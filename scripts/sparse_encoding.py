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
    ax.set_title(title, fontsize=10)
    ax.axis('off')


# -------- Sparse encoding utilities --------
def orthogonalize_against_set(target: torch.Tensor, others: torch.Tensor) -> torch.Tensor:
    """
    target: [1, d], others: [K, d], all assumed on the same device.
    Returns a normalized vector orthogonalized to 'others'.
    """
    if others is None or others.numel() == 0:
        return F.normalize(target, dim=-1)
    B = F.normalize(others, dim=-1)  # [K, d]
    coeff = (target @ B.t())         # [1, K]
    target_res = target - coeff @ B  # [1, d]
    return F.normalize(target_res, dim=-1)


def topk_sparsify(vec: torch.Tensor, k: int) -> torch.Tensor:
    """
    vec: [1, d] normalized; keep top-|value| k dims, renormalize.
    """
    d = vec.shape[-1]
    k = max(1, min(int(k), d))
    _, idx = torch.topk(vec.abs(), k, dim=-1)
    mask = torch.zeros_like(vec).scatter_(-1, idx, 1.0)
    sparse = vec * mask
    return F.normalize(sparse, dim=-1)

def build_dictionary_from_prompts_and_wordlist(
    original_1x: torch.Tensor,
    this_index: int,
    text_emb_all: torch.Tensor,
    tokenizer,
    model,
    wordlist_map: dict,
    prompt_text: str,
    device: torch.device
) -> torch.Tensor:
    """
    Build dictionary D by concatenating other prompt embeddings and external wordlist
    neighbors for the current prompt's extracted keyword.
    Returns D: [K, d] (may be empty if nothing available).
    """
    parts = []
    if this_index > 0:
        parts.append(text_emb_all[:this_index])
    if this_index + 1 < text_emb_all.shape[0]:
        parts.append(text_emb_all[this_index+1:])
    tokens = re.findall(r'[a-z]+', prompt_text.lower())
    key = tokens[-1] if len(tokens) > 0 else ''
    if key and key in wordlist_map and isinstance(wordlist_map[key], list):
        wl = [w for w in wordlist_map[key] if isinstance(w, str) and len(w.strip()) > 0]
        if len(wl) > 0:
            ext_emb = build_wordlist_neighbors_embedding(tokenizer, model, wl, device)
            if ext_emb is not None and ext_emb.numel() > 0:
                parts.append(ext_emb)
    if len(parts) == 0:
        return original_1x.new_zeros((0, original_1x.shape[-1]))
    D = torch.cat(parts, dim=0)
    D = F.normalize(D, dim=-1)
    # Remove any near-duplicate atoms to the target itself
    sim = (D @ original_1x.t()).squeeze(-1).abs()
    keep = sim < 0.999  # drop atoms identical to x
    D = D[keep]
    return D

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
        paths += list_images(args.dataset_root, limit=max(1, args.num_per_class * 2))
    if len(paths) == 0:
        raise RuntimeError(f'No images found under {args.dataset_root}')

    # Process each image: build a grid per image with rows=prompts, cols=len(selected types)
    types_selected = args.sparse_encoding_type or ['original']
    cols = len(types_selected)

    for pth in paths:
        try:
            base_img = Image.open(pth).convert('RGB')
        except Exception:
            continue

        img_t = safe_preprocess(base_img, image_size=args.image_size).unsqueeze(0).to(device)

        fig, axes = plt.subplots(nrows=len(args.prompts), ncols=cols, figsize=(3 * cols, 3 * len(args.prompts)))
        if len(args.prompts) == 1:
            axes = [axes]

        for r, prompt in enumerate(args.prompts):
            original_1x = text_emb_all[r:r+1]  # [1, d]

            maps_for_row: List[Tuple[str, torch.Tensor]] = []

            for c, tname in enumerate(types_selected):
                if tname == 'original':
                    emb_1x = original_1x
                elif tname == 'sparse_residual':
                    # Build dictionary from other prompts + external neighbors (independent of 'word_list' selection)
                    parts = []
                    if r > 0:
                        parts.append(text_emb_all[:r])
                    if r + 1 < text_emb_all.shape[0]:
                        parts.append(text_emb_all[r+1:])
                    tokens = re.findall(r'[a-z]+', prompt.lower())
                    key = tokens[-1] if len(tokens) > 0 else ''
                    wl = external_neighbors_getter(key) if key else []
                    if len(wl) > 0:
                        ext_emb = build_wordlist_neighbors_embedding(tokenizer, model, wl, device)
                        if ext_emb is not None and ext_emb.numel() > 0:
                            parts.append(ext_emb)
                    D = torch.cat(parts, dim=0) if len(parts) > 0 else original_1x.new_zeros((0, original_1x.shape[-1]))
                    if D.numel() > 0:
                        D = F.normalize(D, dim=-1)
                        sim = (D @ original_1x.t()).squeeze(-1).abs()
                        keep = sim < 0.999
                        D = D[keep]
                    emb_1x = omp_sparse_residual(original_1x, D, max_atoms=args.residual_atoms)
                else:
                    emb_1x = original_1x

                heat = compute_map_for_embedding(model, img_t, emb_1x)  # [H, W]
                maps_for_row.append((tname, heat))

            # Plot the row
            for c, (label, heat) in enumerate(maps_for_row):
                title = f'{prompt} - {label}'
                overlay(axes[r][c] if cols > 1 else axes[r], base_img, heat, title=title, alpha=args.overlay_alpha)

        plt.tight_layout()
        base = os.path.splitext(os.path.basename(pth))[0]
        out_name = f'{sanitize(base)}_sparse_encoding.png'
        out_path = os.path.join(args.output_dir, out_name)
        plt.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f'Saved: {out_path}')


if __name__ == '__main__':
    main()


