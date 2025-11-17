#!/usr/bin/env python
import argparse
import os
import re
from typing import List, Tuple

from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
import matplotlib.pyplot as plt
import open_clip

from legrad import LeWrapper


CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def sanitize(name: str) -> str:
    import re as _re
    s = name.strip().lower()
    s = _re.sub(r'[^a-z0-9]+', '_', s)
    s = _re.sub(r'_+', '_', s).strip('_')
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
    if limit > 0:
        return entries[:limit]
    return entries


def overlay(ax, base_img: Image.Image, heat_01: torch.Tensor, title: str, alpha: float = 0.6):
    # heat_01: [H, W] float in [0, 1]
    H, W = heat_01.shape
    base_resized = base_img.resize((W, H), Image.BICUBIC).convert("RGB")
    ax.imshow(base_resized)
    ax.imshow(heat_01.detach().cpu().numpy(), cmap='jet', alpha=alpha, vmin=0.0, vmax=1.0)
    ax.set_title(title, fontsize=9, pad=10)
    ax.axis('off')


def omp_sparse_residual(x_1x: torch.Tensor, D: torch.Tensor, max_atoms: int = 8, tol: float = 1e-6) -> torch.Tensor:
    """
    Orthogonal Matching Pursuit residual.
    x_1x: [1, d] (L2-normalized)
    D: [K, d] atoms (L2-normalized)
    Returns r (L2-normalized): [1, d]
    """
    if D is None or D.numel() == 0:
        return F.normalize(x_1x, dim=-1)
    x = x_1x.clone()  # [1, d]
    K = D.shape[0]
    max_atoms = int(max(1, min(max_atoms, K)))
    selected = []
    r = x.clone()
    for _ in range(max_atoms):
        c = (r @ D.t()).squeeze(0)  # [K]
        c_abs = c.abs()
        if len(selected) > 0:
            c_abs[selected] = -1.0
        idx = int(torch.argmax(c_abs).item())
        if c_abs[idx].item() <= tol:
            break
        selected.append(idx)
        D_S = D[selected, :]
        G = D_S @ D_S.t()
        b = (D_S @ x.t())
        I = torch.eye(G.shape[0], device=G.device, dtype=G.dtype)
        s = torch.linalg.solve(G + 1e-6 * I, b)  # [t,1]
        x_hat = (s.t() @ D_S).to(x.dtype)  # [1, d]
        r = (x - x_hat)
        if float(torch.norm(r)) <= tol:
            break
    if torch.norm(r) <= tol:
        return F.normalize(x, dim=-1)
    return F.normalize(r, dim=-1)


def compute_map_for_embedding(model: LeWrapper, image: torch.Tensor, text_emb_1x: torch.Tensor) -> torch.Tensor:
    """
    text_emb_1x: [1, d] normalized
    Returns heatmap [H, W] in [0,1]
    """
    with torch.enable_grad():
        logits = model.compute_legrad(image=image, text_embedding=text_emb_1x)  # [1, 1, H, W]
    logits = logits[0, 0]
    logits = logits.clamp(0, 1).detach().cpu()
    return logits


def main():
    parser = argparse.ArgumentParser(description='Sparse text encodings (simple): compare original vs prompts-only OMP residual (atoms=8).')
    parser.add_argument('--dataset_root', type=str, required=True, help='Root with images (flat folder or Cat/Dog).')
    parser.add_argument('--prompts', type=str, nargs='*', default=['a photo of a dog.', 'a photo of a cat.'])
    parser.add_argument('--image_size', type=int, default=448)
    parser.add_argument('--model_name', type=str, default='ViT-B-16')
    parser.add_argument('--pretrained', type=str, default='laion2b_s34b_b88k')
    parser.add_argument('--output_dir', type=str, default='outputs/sparse_simple')
    parser.add_argument('--overlay_alpha', type=float, default=0.6)
    parser.add_argument('--max_images', type=int, default=0, help='0=all in flat; per-class fallback otherwise.')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, _, _ = open_clip.create_model_and_transforms(model_name=args.model_name,
                                                        pretrained=args.pretrained,
                                                        device=device)
    tokenizer = open_clip.get_tokenizer(model_name=args.model_name)
    model.eval()
    model = LeWrapper(model, layer_index=0)

    # Text embeddings for prompts
    tok = tokenizer(args.prompts).to(device)
    with torch.no_grad():
        text_emb_all = model.encode_text(tok, normalize=True)  # [P, d]

    # Collect images
    paths = []
    cat_dir = os.path.join(args.dataset_root, 'Cat')
    dog_dir = os.path.join(args.dataset_root, 'Dog')
    if os.path.isdir(cat_dir) and os.path.isdir(dog_dir):
        paths += list_images(cat_dir, limit=args.max_images if args.max_images > 0 else 0)
        paths += list_images(dog_dir, limit=args.max_images if args.max_images > 0 else 0)
    else:
        paths += list_images(args.dataset_root, limit=args.max_images if args.max_images > 0 else 0)
    if len(paths) == 0:
        raise RuntimeError(f'No images found under {args.dataset_root}')
    if args.max_images > 0:
        paths = paths[:args.max_images]

    # Variants: original and prompts-only sparse residual (atoms=8)
    variants: List[Tuple[str, dict]] = [
        ('original', {'mode': 'original'}),
        ('residual@8(prompts_only)', {'mode': 'sparse_residual', 'atoms': 8}),
    ]
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

            # Build dictionary from other prompts only
            parts = []
            if r > 0:
                parts.append(text_emb_all[:r])
            if r + 1 < text_emb_all.shape[0]:
                parts.append(text_emb_all[r+1:])
            D = torch.cat(parts, dim=0) if len(parts) > 0 else original_1x.new_zeros((0, original_1x.shape[-1]))
            if D.numel() > 0:
                D = F.normalize(D, dim=-1)
                sim = (D @ original_1x.t()).squeeze(-1).abs()
                keep = sim < 0.999
                D = D[keep]
            print(f'[D] prompt="{prompt}" rows={0 if D.numel()==0 else D.shape[0]} atoms=8')

            maps_for_row: List[Tuple[str, torch.Tensor]] = []
            for c, (vlabel, vcfg) in enumerate(variants):
                if vcfg.get('mode') == 'original':
                    emb_1x = original_1x
                else:
                    emb_1x = omp_sparse_residual(original_1x, D, max_atoms=8)
                heat = compute_map_for_embedding(model, img_t, emb_1x)
                maps_for_row.append((vlabel, heat))

            for c, (label, heat) in enumerate(maps_for_row):
                title = f'{prompt} - {label}'
                overlay(axes[r][c] if cols > 1 else axes[r], base_img, heat, title=title, alpha=args.overlay_alpha)

        plt.tight_layout()
        try:
            plt.subplots_adjust(hspace=1.0, wspace=1.0)
        except Exception:
            pass
        base = os.path.splitext(os.path.basename(pth))[0]
        out_name = f'{sanitize(base)}_sparse_simple.png'
        out_path = os.path.join(args.output_dir, out_name)
        plt.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f'Saved: {out_path}')


if __name__ == '__main__':
    main()


000