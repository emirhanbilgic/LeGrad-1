#!/usr/bin/env python
import argparse
import os
import re
import json
from typing import List, Tuple

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
    parser = argparse.ArgumentParser(description='Sparse text encodings for LeGrad: word-list orthogonalization and hard masking.')
    parser.add_argument('--dataset_root', type=str, required=True, help='Root with images (expects Cat/ and Dog/ if present).')
    parser.add_argument('--num_per_class', type=int, default=4, help='Images per class to sample (if Cat/Dog exist).')
    parser.add_argument('--image_size', type=int, default=448)
    parser.add_argument('--model_name', type=str, default='ViT-B-16')
    parser.add_argument('--pretrained', type=str, default='laion2b_s34b_b88k')
    parser.add_argument('--prompts', type=str, nargs='*', default=['a photo of a dog.', 'a photo of a cat.'])
    parser.add_argument('--sparse_encoding_type', type=str, nargs='*',
                        default=['original'],
                        choices=['original', 'word_list', 'hard'],
                        help='Select one or more types; default tries original only.')
    parser.add_argument('--wordlist_path', type=str, default='resources/wordlist_neighbors.json',
                        help='Path to JSON mapping from keyword to list of neighbor words for orthogonalization.')
    parser.add_argument('--topk', type=int, default=64, help='k for hard masking.')
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

    # Load external wordlist neighbors (JSON)
    wordlist_map = {}
    if args.wordlist_path and os.path.isfile(args.wordlist_path):
        try:
            with open(args.wordlist_path, 'r') as f:
                wordlist_map = json.load(f)
            if not isinstance(wordlist_map, dict):
                wordlist_map = {}
        except Exception:
            wordlist_map = {}
    else:
        # If file is missing, proceed without external neighbors
        wordlist_map = {}

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
            # Build neighbors for this prompt (other prompts + external wordlist)
            neighbors = []
            if 'word_list' in types_selected:
                # other prompts
                if r > 0:
                    neighbors.append(text_emb_all[:r])
                if r + 1 < text_emb_all.shape[0]:
                    neighbors.append(text_emb_all[r+1:])
                # external wordlist for keyword extracted from prompt
                # extract keyword as last alphabetic token
                tokens = re.findall(r'[a-z]+', prompt.lower())
                key = tokens[-1] if len(tokens) > 0 else ''
                wl = []
                if key and key in wordlist_map and isinstance(wordlist_map[key], list):
                    wl = [w for w in wordlist_map[key] if isinstance(w, str) and len(w.strip()) > 0]
                if len(wl) > 0:
                    ext_emb = build_wordlist_neighbors_embedding(tokenizer, model, wl, device)
                    if ext_emb is not None and ext_emb.numel() > 0:
                        neighbors.append(ext_emb)
            neighbors_t = torch.cat(neighbors, dim=0) if len(neighbors) > 0 else None

            maps_for_row: List[Tuple[str, torch.Tensor]] = []

            for c, tname in enumerate(types_selected):
                if tname == 'original':
                    emb_1x = original_1x
                elif tname == 'word_list':
                    emb_1x = orthogonalize_against_set(original_1x, neighbors_t)
                elif tname == 'hard':
                    emb_1x = topk_sparsify(original_1x, k=args.topk)
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


