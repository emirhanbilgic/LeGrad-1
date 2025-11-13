#!/usr/bin/env python
import argparse
import os
import re
import json
from typing import List, Tuple
import requests

from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
import matplotlib.pyplot as plt
from einops import rearrange
import open_clip

from legrad import LeWrapper


CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def sanitize(name: str) -> str:
    s = name.strip().lower()
    s = re.sub(r'[^a-z0-9]+', '_', s)
    s = re.sub(r'_+', '_', s).strip('_')
    return s or 'x'


def pil_to_tensor_no_numpy(img: Image.Image) -> torch.Tensor:
    img = img.convert("RGB")
    w, h = img.size
    byte_data = img.tobytes()
    t = torch.tensor(list(byte_data), dtype=torch.uint8)
    t = t.view(h, w, 3).permute(2, 0, 1)
    return t


def safe_preprocess(img: Image.Image, image_size: int = 448) -> torch.Tensor:
    t = pil_to_tensor_no_numpy(img)
    t = TF.resize(t, [image_size, image_size], interpolation=InterpolationMode.BICUBIC, antialias=True)
    t = TF.center_crop(t, [image_size, image_size])
    t = t.float() / 255.0
    mean = torch.tensor(CLIP_MEAN).view(3, 1, 1)
    std = torch.tensor(CLIP_STD).view(3, 1, 1)
    t = (t - mean) / std
    return t


def list_images(folder: str, limit: int, seed: int = 42) -> List[str]:
    import random
    entries = []
    if not os.path.isdir(folder):
        return entries
    for name in sorted(os.listdir(folder)):
        path = os.path.join(folder, name)
        if os.path.isfile(path):
            ext = name.lower().rsplit(".", 1)[-1]
            if ext in {"jpg", "jpeg", "png", "bmp", "webp"}:
                entries.append(path)
    random.Random(seed).shuffle(entries)
    return entries[:limit]


def min_max_batch(x: torch.Tensor) -> torch.Tensor:
    # x: [1, P, H, W] -> min-max per [P, H, W]
    B, P = x.shape[:2]
    x_ = x.reshape(B, P, -1)
    minv = x_.min(dim=-1, keepdim=True)[0]
    maxv = x_.max(dim=-1, keepdim=True)[0]
    x_ = (x_ - minv) / (maxv - minv + 1e-6)
    return x_.reshape_as(x)


def overlay(ax, base_img: Image.Image, heat_01: torch.Tensor, title: str, alpha: float = 0.6):
    # heat_01: [H, W] float in [0, 1]
    H, W = heat_01.shape
    base_resized = base_img.resize((W, H), Image.BICUBIC).convert("RGB")
    ax.imshow(base_resized)
    ax.imshow(heat_01.detach().cpu().numpy(), cmap='jet', alpha=alpha, vmin=0.0, vmax=1.0)
    ax.set_title(title, fontsize=9, pad=10)
    ax.axis('off')


# -------- Sparse encoding utilities --------

def omp_sparse_residual(x_1x: torch.Tensor, D: torch.Tensor, max_atoms: int = 8, tol: float = 1e-6) -> torch.Tensor:
    """
    Simple Orthogonal Matching Pursuit to compute sparse coding residual without training.
    x_1x: [1, d], assumed L2-normalized
    D: [K, d], atom rows, L2-normalized
    Returns residual r (L2-normalized): [1, d]
    """
    if D is None or D.numel() == 0:
        return F.normalize(x_1x, dim=-1)
    x = x_1x.clone()  # [1, d]
    K = D.shape[0]
    max_atoms = int(max(1, min(max_atoms, K)))
    selected = []
    r = x.clone()  # residual starts as x
    for _ in range(max_atoms):
        # correlations with residual
        c = (r @ D.t()).squeeze(0)  # [K]
        c_abs = c.abs()
        # mask already selected
        if len(selected) > 0:
            c_abs[selected] = -1.0
        idx = int(torch.argmax(c_abs).item())
        if c_abs[idx].item() <= tol:
            break
        selected.append(idx)
        # Solve least squares on selected atoms: s = argmin ||x - s^T D_S||^2
        D_S = D[selected, :]  # [t, d]
        G = D_S @ D_S.t()     # [t, t]
        b = (D_S @ x.t())     # [t, 1]
        # Regularize G slightly for stability
        I = torch.eye(G.shape[0], device=G.device, dtype=G.dtype)
        s = torch.linalg.solve(G + 1e-6 * I, b)  # [t,1]
        x_hat = (s.t() @ D_S).to(x.dtype)  # [1, d]
        r = (x - x_hat)
        # Early stop if residual very small
        if float(torch.norm(r)) <= tol:
            break
    # Return normalized residual (fallback to x if degenerate)
    if torch.norm(r) <= tol:
        return F.normalize(x, dim=-1)
    return F.normalize(r, dim=-1)


def build_wordlist_neighbors_embedding(tokenizer, model, words: List[str], device: torch.device) -> torch.Tensor:
    """
    Encode additional neighbor words into text embeddings.
    If words is empty, returns None.
    """
    if words is None or len(words) == 0:
        return None
    # Simple pattern: use the raw words as prompts
    tok = tokenizer(words).to(device)
    with torch.no_grad():
        emb = model.encode_text(tok, normalize=True)  # [K, d]
    return emb

def filter_neighbors_by_clip_similarity(
    original_1x: torch.Tensor,
    neighbor_words: List[str],
    neighbor_embs: torch.Tensor,
    min_sim: float,
    max_sim: float,
    topk: int,
    diverse: bool = False
) -> Tuple[List[str], torch.Tensor, torch.Tensor]:
    """
    Filter neighbor strings using CLIP cosine similarity band to the original embedding.
    Optionally select a diverse subset via greedy farthest-first.
    Returns (filtered_words, filtered_embs, filtered_sims).
    """
    if neighbor_embs is None or neighbor_embs.numel() == 0 or len(neighbor_words) == 0:
        return [], neighbor_embs.new_zeros((0, original_1x.shape[-1])), neighbor_embs.new_zeros((0,))
    sims = (neighbor_embs @ original_1x.t()).squeeze(-1)  # [K]
    keep_mask = torch.ones_like(sims, dtype=torch.bool)
    if min_sim is not None:
        keep_mask &= sims >= float(min_sim)
    if max_sim is not None:
        keep_mask &= sims <= float(max_sim)
    idxs = torch.nonzero(keep_mask, as_tuple=False).squeeze(-1).tolist()
    if len(idxs) == 0:
        return [], neighbor_embs.new_zeros((0, original_1x.shape[-1])), neighbor_embs.new_zeros((0,))
    words_kept = [neighbor_words[i] for i in idxs]
    embs_kept = neighbor_embs[idxs]
    sims_kept = sims[idxs]
    if topk is not None and int(topk) > 0 and len(words_kept) > int(topk):
        if diverse:
            # Greedy farthest-first selection in embedding space
            selected = []
            candidates = list(range(len(words_kept)))
            # seed with the item closest to the band center (to avoid extremes)
            center = 0.5 * (float(min_sim if min_sim is not None else -1.0) + float(max_sim if max_sim is not None else 1.0))
            seed = min(candidates, key=lambda i: abs(float(sims_kept[i]) - center))
            selected.append(seed)
            candidates.remove(seed)
            while len(selected) < int(topk) and len(candidates) > 0:
                # pick candidate with minimum average cosine similarity to selected (i.e., most diverse)
                sel_mat = embs_kept[selected]  # [t, d]
                best_cand = None
                best_score = 1e9
                for j in candidates:
                    cj = embs_kept[j:j+1]  # [1, d]
                    cs = (cj @ sel_mat.t()).abs().mean().item()
                    if cs < best_score:
                        best_score = cs
                        best_cand = j
                selected.append(best_cand)
                candidates.remove(best_cand)
            mask = torch.zeros(len(words_kept), dtype=torch.bool)
            for i in selected:
                mask[i] = True
            embs_kept = embs_kept[mask]
            sims_kept = sims_kept[mask]
            words_kept = [w for m, w in zip(mask.tolist(), words_kept) if m]
        else:
            # keep topk by descending similarity within band
            order = torch.argsort(sims_kept, descending=True)[: int(topk)]
            embs_kept = embs_kept[order]
            sims_kept = sims_kept[order]
            words_kept = [words_kept[i] for i in order.tolist()]
    return words_kept, embs_kept, sims_kept

def wordnet_neighbors(keyword: str, limit_per_relation: int = 8) -> List[str]:
    """
    Collect WordNet neighbors: synonyms, hypernyms, hyponyms, and co-hyponyms (siblings).
    Returns a deduplicated, lowercase list excluding the keyword.
    """
    try:
        import nltk  # type: ignore
        from nltk.corpus import wordnet as wn  # type: ignore
    except Exception:
        return []
    out = []
    seen = set()
    key_low = keyword.lower()
    synsets = wn.synsets(keyword, pos=wn.NOUN)
    for s in synsets[:limit_per_relation]:
        # synonyms
        for l in s.lemmas()[:limit_per_relation]:
            name = l.name().replace('_', ' ').lower()
            if name != key_low and name not in seen:
                out.append(name); seen.add(name)
        # hypernyms
        for h in s.hypernyms()[:limit_per_relation]:
            for l in h.lemmas()[:limit_per_relation]:
                name = l.name().replace('_', ' ').lower()
                if name != key_low and name not in seen:
                    out.append(name); seen.add(name)
        # hyponyms
        for h in s.hyponyms()[:limit_per_relation]:
            for l in h.lemmas()[:limit_per_relation]:
                name = l.name().replace('_', ' ').lower()
                if name != key_low and name not in seen:
                    out.append(name); seen.add(name)
        # co-hyponyms (siblings)
        for h in s.hypernyms()[:limit_per_relation]:
            for sib in h.hyponyms()[:limit_per_relation]:
                for l in sib.lemmas()[:limit_per_relation]:
                    name = l.name().replace('_', ' ').lower()
                    if name != key_low and name not in seen:
                        out.append(name); seen.add(name)
    return out[: max(1, limit_per_relation * 3)]

def wordnet_neighbors_configured(
    keyword: str,
    use_synonyms: bool,
    use_hypernyms: bool,
    use_hyponyms: bool,
    use_siblings: bool,
    limit_per_relation: int = 8
) -> List[str]:
    """
    Configurable WordNet neighbors. Enable/disable relations via flags.
    """
    try:
        import nltk  # type: ignore
        from nltk.corpus import wordnet as wn  # type: ignore
    except Exception:
        return []
    out = []
    seen = set()
    key_low = keyword.lower()
    synsets = wn.synsets(keyword, pos=wn.NOUN)
    for s in synsets[:limit_per_relation]:
        if use_synonyms:
            for l in s.lemmas()[:limit_per_relation]:
                name = l.name().replace('_', ' ').lower()
                if name != key_low and name not in seen:
                    out.append(name); seen.add(name)
        if use_hypernyms:
            for h in s.hypernyms()[:limit_per_relation]:
                for l in h.lemmas()[:limit_per_relation]:
                    name = l.name().replace('_', ' ').lower()
                    if name != key_low and name not in seen:
                        out.append(name); seen.add(name)
        if use_hyponyms:
            for h in s.hyponyms()[:limit_per_relation]:
                for l in h.lemmas()[:limit_per_relation]:
                    name = l.name().replace('_', ' ').lower()
                    if name != key_low and name not in seen:
                        out.append(name); seen.add(name)
        if use_siblings:
            for h in s.hypernyms()[:limit_per_relation]:
                for sib in h.hyponyms()[:limit_per_relation]:
                    for l in sib.lemmas()[:limit_per_relation]:
                        name = l.name().replace('_', ' ').lower()
                        if name != key_low and name not in seen:
                            out.append(name); seen.add(name)
    # Cap list size reasonably
    return out[: max(1, limit_per_relation * 3)]


def compute_map_for_embedding(model: LeWrapper, image: torch.Tensor, text_emb_1x: torch.Tensor) -> torch.Tensor:
    """
    text_emb_1x: [1, d] normalized
    Returns heatmap [H, W] in [0,1]
    """
    with torch.enable_grad():
        logits = model.compute_legrad(image=image, text_embedding=text_emb_1x)  # [1, 1, H, W]
    logits = logits[0, 0]
    # Already min-max normalized by compute_legrad for CLIP; ensure clamp
    logits = logits.clamp(0, 1).detach().cpu()
    return logits


def main():
    parser = argparse.ArgumentParser(description='Sparse text encodings for LeGrad: original and sparse residual modes.')
    parser.add_argument('--dataset_root', type=str, required=True, help='Root with images (expects Cat/ and Dog/ if present).')
    parser.add_argument('--num_per_class', type=int, default=4, help='Images per class to sample (if Cat/Dog exist).')
    parser.add_argument('--image_size', type=int, default=448)
    parser.add_argument('--model_name', type=str, default='ViT-B-16')
    parser.add_argument('--pretrained', type=str, default='laion2b_s34b_b88k')
    parser.add_argument('--prompts', type=str, nargs='*', default=['a photo of a dog.', 'a photo of a cat.'])
    parser.add_argument('--sparse_encoding_type', type=str, nargs='*',
                        default=['original'],
                        choices=['original', 'sparse_residual'],
                        help='Select one or more types; default tries original only.')
    parser.add_argument('--wordlist_source', type=str, default='json',
                        choices=['json', 'url', 'wordnet'],
                        help='Source for neighbor words: local JSON, URL JSON, or WordNet.')
    parser.add_argument('--wordlist_path', type=str, default='resources/wordlist_neighbors.json',
                        help='When --wordlist_source=json, path to JSON mapping from keyword to neighbor list.')
    parser.add_argument('--wordlist_url', type=str, default='',
                        help='When --wordlist_source=url, URL to JSON mapping from keyword to neighbor list.')
    parser.add_argument('--residual_atoms', type=int, default=8, help='Max atoms for OMP residual.')
    parser.add_argument('--wn_use_synonyms', type=int, default=0, help='WordNet: include synonyms (0/1).')
    parser.add_argument('--wn_use_hypernyms', type=int, default=0, help='WordNet: include hypernyms (0/1).')
    parser.add_argument('--wn_use_hyponyms', type=int, default=0, help='WordNet: include hyponyms (0/1).')
    parser.add_argument('--wn_use_siblings', type=int, default=1, help='WordNet: include co-hyponyms/siblings (0/1).')
    parser.add_argument('--output_dir', type=str, default='outputs/sparse_encoding')
    parser.add_argument('--overlay_alpha', type=float, default=0.6)
    parser.add_argument('--benchmark', type=int, default=0,
                        help='Enable benchmark grid over multiple WordNet configs and atom counts (0/1).')
    parser.add_argument('--atom_grid', type=int, nargs='*', default=[8, 16, 24, 32],
                        help='Atom counts to test when --benchmark=1.')
    parser.add_argument('--wn_configs', type=str, nargs='*', default=['none', 'siblings', 'siblings+hyponyms', 'siblings_clipband', 'siblings_clipband_diverse'],
                        help='WordNet configs to test when --benchmark=1. '
                             'Choices include: siblings, siblings+hyponyms, synonyms, hypernyms, hyponyms, '
                             'siblings+synonyms, siblings_clipband, siblings_clipband_diverse, none (prompts only).')
    parser.add_argument('--max_images', type=int, default=0,
                        help='Cap total images processed. 0 means auto (per-class or 2*num_per_class for flat).')
    parser.add_argument('--neighbor_min_sim', type=float, default=-1.0,
                        help='Minimum CLIP cosine similarity band for neighbor filtering; <0 disables.')
    parser.add_argument('--neighbor_max_sim', type=float, default=2.0,
                        help='Maximum CLIP cosine similarity band for neighbor filtering; >1 disables.')
    parser.add_argument('--neighbors_topk', type=int, default=0,
                        help='Top-K neighbors to keep after CLIP-band filtering; 0 disables.')
    parser.add_argument('--neighbors_diverse', type=int, default=0,
                        help='If 1, use farthest-first selection for diversity when topk > 0.')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, _, _ = open_clip.create_model_and_transforms(model_name=args.model_name,
                                                        pretrained=args.pretrained,
                                                        device=device)
    tokenizer = open_clip.get_tokenizer(model_name=args.model_name)
    model.eval()
    # Wrap with LeGrad, include all layers
    model = LeWrapper(model, layer_index=0)

    # Text embeddings for prompts
    tok = tokenizer(args.prompts).to(device)
    with torch.no_grad():
        text_emb_all = model.encode_text(tok, normalize=True)  # [P, d]

    # Prepare external neighbor source getter
    wordlist_map = {}
    if args.wordlist_source == 'json':
        if args.wordlist_path and os.path.isfile(args.wordlist_path):
            try:
                with open(args.wordlist_path, 'r') as f:
                    wordlist_map = json.load(f)
                if not isinstance(wordlist_map, dict):
                    wordlist_map = {}
            except Exception:
                wordlist_map = {}
        def external_neighbors_getter(key: str) -> List[str]:
            if key and key in wordlist_map and isinstance(wordlist_map[key], list):
                return [w for w in wordlist_map[key] if isinstance(w, str) and len(w.strip()) > 0]
            return []
    elif args.wordlist_source == 'url':
        url_map = {}
        if args.wordlist_url:
            try:
                resp = requests.get(args.wordlist_url, timeout=10)
                if resp.ok:
                    url_map = resp.json()
                    if not isinstance(url_map, dict):
                        url_map = {}
            except Exception:
                url_map = {}
        def external_neighbors_getter(key: str) -> List[str]:
            if key and key in url_map and isinstance(url_map[key], list):
                return [w for w in url_map[key] if isinstance(w, str) and len(w.strip()) > 0]
            return []
    else:
        # WordNet dynamic neighbors
        def external_neighbors_getter(key: str) -> List[str]:
            return wordnet_neighbors_configured(
                key,
                use_synonyms=bool(args.wn_use_synonyms),
                use_hypernyms=bool(args.wn_use_hypernyms),
                use_hyponyms=bool(args.wn_use_hyponyms),
                use_siblings=bool(args.wn_use_siblings),
                limit_per_relation=8
            )

    # Collect images
    paths = []
    cat_dir = os.path.join(args.dataset_root, 'Cat')
    dog_dir = os.path.join(args.dataset_root, 'Dog')
    if os.path.isdir(cat_dir) and os.path.isdir(dog_dir):
        paths += list_images(cat_dir, limit=args.num_per_class)
        paths += list_images(dog_dir, limit=args.num_per_class)
    else:
        # Fallback: flat directory listing
        flat_limit = args.max_images if args.max_images > 0 else max(1, args.num_per_class * 2)
        paths += list_images(args.dataset_root, limit=flat_limit)
    if len(paths) == 0:
        raise RuntimeError(f'No images found under {args.dataset_root}')
    # If a hard cap is set, enforce it
    if args.max_images > 0:
        paths = paths[:args.max_images]

    # Helpers to construct benchmark variants
    def parse_wn_config_name(name: str) -> Tuple[bool, bool, bool, bool]:
        n = (name or '').strip().lower()
        if n in {'none', 'prompts_only', 'others_only', 'no_neighbors'}:
            return False, False, False, False
        use_synonyms = ('synonym' in n)
        use_hypernyms = ('hypernym' in n)
        use_hyponyms = ('hyponym' in n)
        use_siblings = ('sibling' in n) or (n == 'siblings') or (n == 'cohyponyms') or (n == 'co-hyponyms')
        return bool(use_synonyms), bool(use_hypernyms), bool(use_hyponyms), bool(use_siblings)
    def parse_variant_filters(name: str):
        n = (name or '').strip().lower()
        if 'clipband' in n:
            # Use a moderately specific band and topk for disentanglement without killing coverage
            clip_filter = {
                'min_sim': 0.2,
                'max_sim': 0.6,
                'topk': 16,
                'diverse': True if 'diverse' in n else False
            }
        else:
            clip_filter = None
        return clip_filter
    def variant_disables_neighbors(name: str) -> bool:
        n = (name or '').strip().lower()
        return n in {'none', 'prompts_only', 'others_only', 'no_neighbors'}

    # Process each image: build a grid per image with rows=prompts, cols=len(variants)
    types_selected = args.sparse_encoding_type or ['original']
    variants: List[Tuple[str, dict]] = []
    variants.append(('original', {'mode': 'original'}))
    if 'sparse_residual' in types_selected:
        if args.benchmark:
            # Expand into multiple configs x atoms
            for cfg_name in args.wn_configs:
                flags = parse_wn_config_name(cfg_name)
                clip_filter = parse_variant_filters(cfg_name)
                for k in (args.atom_grid or [args.residual_atoms]):
                    label = f'sparse_residual:{cfg_name}@{int(k)}'
                    variants.append((
                        label,
                        {
                            'mode': 'sparse_residual',
                            'wn_cfg_name': cfg_name,
                            'wn_flags': {
                                'use_synonyms': flags[0],
                                'use_hypernyms': flags[1],
                                'use_hyponyms': flags[2],
                                'use_siblings': flags[3],
                            },
                            'clip_filter': clip_filter,
                            'disable_neighbors': variant_disables_neighbors(cfg_name),
                            'atoms': int(k),
                        }
                    ))
        else:
            # Single configuration driven by top-level args
            variants.append((
                'sparse_residual',
                {
                    'mode': 'sparse_residual',
                    'wn_cfg_name': 'args',
                    'wn_flags': {
                        'use_synonyms': bool(args.wn_use_synonyms),
                        'use_hypernyms': bool(args.wn_use_hypernyms),
                        'use_hyponyms': bool(args.wn_use_hyponyms),
                        'use_siblings': bool(args.wn_use_siblings),
                    },
                    'clip_filter': {
                        'min_sim': None if args.neighbor_min_sim < 0 else float(args.neighbor_min_sim),
                        'max_sim': None if args.neighbor_max_sim > 1.0 else float(args.neighbor_max_sim),
                        'topk': int(args.neighbors_topk) if args.neighbors_topk > 0 else 0,
                        'diverse': bool(args.neighbors_diverse),
                    } if (args.neighbor_min_sim >= 0 or args.neighbor_max_sim <= 1.0 or args.neighbors_topk > 0) else None,
                    'atoms': int(args.residual_atoms),
                }
            ))
    cols = len(variants)

    for pth in paths:
        try:
            base_img = Image.open(pth).convert('RGB')
        except Exception:
            continue

        img_t = safe_preprocess(base_img, image_size=args.image_size).unsqueeze(0).to(device)

        fig, axes = plt.subplots(nrows=len(args.prompts), ncols=cols, figsize=(3.5 * cols, 3.5 * len(args.prompts)))
        if len(args.prompts) == 1:
            axes = [axes]

        for r, prompt in enumerate(args.prompts):
            original_1x = text_emb_all[r:r+1]  # [1, d]

            maps_for_row: List[Tuple[str, torch.Tensor]] = []

            for c, (vlabel, vcfg) in enumerate(variants):
                mode = vcfg.get('mode', 'original')
                if mode == 'original':
                    emb_1x = original_1x
                elif mode == 'sparse_residual':
                    # Build dictionary from other prompts + neighbors
                    parts = []
                    if r > 0:
                        parts.append(text_emb_all[:r])
                    if r + 1 < text_emb_all.shape[0]:
                        parts.append(text_emb_all[r+1:])
                    tokens = re.findall(r'[a-z]+', prompt.lower())
                    key = tokens[-1] if len(tokens) > 0 else ''
                    # Determine neighbors based on benchmark config vs. global args/loader
                    if vcfg.get('disable_neighbors', False):
                        wl = []
                    else:
                        if key:
                            if args.benchmark and vcfg.get('wn_cfg_name') is not None:
                                f = vcfg.get('wn_flags', {})
                                wl = wordnet_neighbors_configured(
                                    key,
                                    use_synonyms=bool(f.get('use_synonyms', False)),
                                    use_hypernyms=bool(f.get('use_hypernyms', False)),
                                    use_hyponyms=bool(f.get('use_hyponyms', False)),
                                    use_siblings=bool(f.get('use_siblings', True)),
                                    limit_per_relation=8
                                )
                            else:
                                wl = external_neighbors_getter(key)
                        else:
                            wl = []
                    # Optional CLIP-band filtering and top-k selection for disentanglement
                    clip_filter = vcfg.get('clip_filter', None)
                    ext_emb = None
                    if len(wl) > 0:
                        # Prepare embeddings to filter by CLIP similarity if requested
                        if clip_filter is not None:
                            # Build emb for all neighbors, then filter
                            all_emb = build_wordlist_neighbors_embedding(tokenizer, model, wl, device)
                            if all_emb is not None and all_emb.numel() > 0:
                                min_sim = clip_filter.get('min_sim', None)
                                max_sim = clip_filter.get('max_sim', None)
                                topk = clip_filter.get('topk', 0)
                                diverse = bool(clip_filter.get('diverse', False))
                                # Treat sentinel defaults as disabled
                                if min_sim is not None and float(min_sim) < 0:
                                    min_sim = None
                                if max_sim is not None and float(max_sim) > 1.0:
                                    max_sim = None
                                wl, all_emb, _ = filter_neighbors_by_clip_similarity(
                                    original_1x, wl, all_emb,
                                    min_sim=min_sim, max_sim=max_sim,
                                    topk=int(topk) if topk is not None else 0,
                                    diverse=diverse
                                )
                                ext_emb = all_emb if all_emb is not None and all_emb.numel() > 0 else None
                        # If not filtered, defer to building embeddings normally
                        if ext_emb is None:
                            ext_emb = build_wordlist_neighbors_embedding(tokenizer, model, wl, device)
                    # Print the created (possibly filtered) wordlist for visibility
                    if key is not None:
                        try:
                            print(f'[neighbors] prompt="{prompt}" key="{key}" variant="{vlabel}" '
                                  f'num={len(wl)}: {wl}')
                        except Exception:
                            pass
                    if len(wl) > 0:
                        if ext_emb is not None and ext_emb.numel() > 0:
                            parts.append(ext_emb)
                    D = torch.cat(parts, dim=0) if len(parts) > 0 else original_1x.new_zeros((0, original_1x.shape[-1]))
                    if D.numel() > 0:
                        D = F.normalize(D, dim=-1)
                        sim = (D @ original_1x.t()).squeeze(-1).abs()
                        keep = sim < 0.999
                        D = D[keep]
                    max_atoms = int(vcfg.get('atoms', args.residual_atoms))
                    emb_1x = omp_sparse_residual(original_1x, D, max_atoms=max_atoms)
                else:
                    emb_1x = original_1x

                heat = compute_map_for_embedding(model, img_t, emb_1x)  # [H, W]
                maps_for_row.append((vlabel, heat))

            # Plot the row
            for c, (label, heat) in enumerate(maps_for_row):
                title = f'{prompt} - {label}'
                overlay(axes[r][c] if cols > 1 else axes[r], base_img, heat, title=title, alpha=args.overlay_alpha)

        plt.tight_layout()
        # Leave more gaps to avoid overlapping titles/labels
        try:
            plt.subplots_adjust(hspace=1.0, wspace=1)
        except Exception:
            pass
        base = os.path.splitext(os.path.basename(pth))[0]
        out_name = f'{sanitize(base)}_sparse_encoding.png'
        out_path = os.path.join(args.output_dir, out_name)
        plt.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f'Saved: {out_path}')


if __name__ == '__main__':
    main()


