#!/usr/bin/env python

import argparse
import os
import random
from io import BytesIO
from typing import List, Tuple
import re

import requests
from PIL import Image
from math import ceil

import torch
import torch.nn.functional as F
from torchvision.transforms import InterpolationMode
import torchvision.transforms.functional as TF
import open_clip
from einops import rearrange
import matplotlib.pyplot as plt

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


def list_images(folder: str, limit: int, seed: int) -> List[str]:
    entries = []
    for name in sorted(os.listdir(folder)):
        path = os.path.join(folder, name)
        if os.path.isfile(path):
            ext = name.lower().rsplit(".", 1)[-1]
            if ext in {"jpg", "jpeg", "png", "bmp", "webp"}:
                entries.append(path)
    random.Random(seed).shuffle(entries)
    return entries[:limit]


@torch.no_grad()
def _grid_wh_from_tokens(num_tokens: int) -> Tuple[int, int]:
    w = int(num_tokens ** 0.5)
    h = w
    return w, h


def compute_per_layer_activation_clip(model: LeWrapper, image: torch.Tensor, text_embedding: torch.Tensor) -> torch.Tensor:
    """
    Returns per-layer scalar activations: shape [num_layers]
    Implementation mirrors compute_legrad_clip but records mean activation per layer before summation.
    """
    assert text_embedding.ndim == 2  # [num_prompts, dim]
    num_prompts = text_embedding.shape[0]

    # Encode image to populate hooks and layer features
    _ = model.encode_image(image)

    blocks_list = list(dict(model.visual.transformer.resblocks.named_children()).values())
    image_features_list = []
    for layer in range(model.starting_depth, len(model.visual.transformer.resblocks)):
        intermediate_feat = model.visual.transformer.resblocks[layer].feat_post_mlp  # [num_patch, batch, dim]
        intermediate_feat = model.visual.ln_post(intermediate_feat.mean(dim=0)) @ model.visual.proj
        intermediate_feat = F.normalize(intermediate_feat, dim=-1)
        image_features_list.append(intermediate_feat)

    num_tokens = blocks_list[-1].feat_post_mlp.shape[0] - 1
    w, h = _grid_wh_from_tokens(num_tokens)

    per_layer_vals = []
    for layer, (blk, img_feat) in enumerate(zip(blocks_list[model.starting_depth:], image_features_list)):
        model.visual.zero_grad()
        sim = text_embedding @ img_feat.transpose(-1, -2)
        one_hot = torch.arange(0, num_prompts, device=text_embedding.device)
        one_hot = F.one_hot(one_hot, num_classes=num_prompts).float()
        one_hot = torch.sum(one_hot * sim)

        attn_map = blocks_list[model.starting_depth + layer].attn.attention_maps  # [(b*h), N, N]
        grad = torch.autograd.grad(one_hot, [attn_map], retain_graph=True, create_graph=True)[0]
        grad = rearrange(grad, '(b h) n m -> b h n m', b=num_prompts)
        grad = torch.clamp(grad, min=0.)

        image_relevance = grad.mean(dim=1).mean(dim=1)[:, 1:]  # [b, N-1]
        expl_map = rearrange(image_relevance, 'b (w h) -> b w h', w=w, h=h)  # no upsample needed for scalar
        # Scalar activation per layer: mean over spatial and prompts
        per_layer_vals.append(expl_map.mean().detach())

    per_layer = torch.stack(per_layer_vals)  # [num_layers]
    return per_layer


def compute_per_layer_head_activation_clip(model: LeWrapper, image: torch.Tensor, text_embedding: torch.Tensor) -> torch.Tensor:
    """
    Returns per-layer, per-head scalar activations: shape [num_layers, num_heads]
    Aggregates by mean over prompts and spatial tokens after ReLU(grad).
    """
    assert text_embedding.ndim == 2  # [num_prompts, dim]
    num_prompts = text_embedding.shape[0]

    _ = model.encode_image(image)

    blocks_list = list(dict(model.visual.transformer.resblocks.named_children()).values())
    image_features_list = []
    for layer in range(model.starting_depth, len(model.visual.transformer.resblocks)):
        intermediate_feat = model.visual.transformer.resblocks[layer].feat_post_mlp  # [num_patch, batch, dim]
        intermediate_feat = model.visual.ln_post(intermediate_feat.mean(dim=0)) @ model.visual.proj
        intermediate_feat = F.normalize(intermediate_feat, dim=-1)
        image_features_list.append(intermediate_feat)

    num_tokens = blocks_list[-1].feat_post_mlp.shape[0] - 1
    w, h = _grid_wh_from_tokens(num_tokens)

    per_layer_head_vals = []
    for layer, (blk, img_feat) in enumerate(zip(blocks_list[model.starting_depth:], image_features_list)):
        model.visual.zero_grad()
        sim = text_embedding @ img_feat.transpose(-1, -2)  # [num_prompts, N]
        one_hot = torch.sum(sim)  # scalar objective across prompts

        attn_map = blocks_list[model.starting_depth + layer].attn.attention_maps  # [(b*h), N, N]
        grad = torch.autograd.grad(one_hot, [attn_map], retain_graph=True, create_graph=True)[0]
        grad = rearrange(grad, '(b h) n m -> b h n m', b=num_prompts)  # [b,h,n,m]
        grad = torch.clamp(grad, min=0.)

        # Average over query dimension, drop CLS on key dimension, then average over tokens -> [b,h]
        head_token_relevance = grad.mean(dim=2)[:, :, 1:]  # [b,h,N-1]
        head_scalar = head_token_relevance.mean(dim=2).mean(dim=0)  # -> [h]
        per_layer_head_vals.append(head_scalar.detach())

    per_layer_head = torch.stack(per_layer_head_vals)  # [L,H]
    return per_layer_head


def tensor_rgb_to_pil_bytes(t: torch.ByteTensor) -> bytes:
    if t.dtype != torch.uint8:
        t = t.to(torch.uint8)
    t = t.clamp(0, 255)
    t_hw3 = t.permute(1, 2, 0).contiguous().view(-1)
    return bytes(t_hw3.tolist())


def overlay_heat_to_pil(base_img: Image.Image, heat_01: torch.Tensor, alpha: float = 0.6) -> Image.Image:
    """
    base_img: PIL RGB, any size
    heat_01: torch.float [H, W] in [0,1]
    Returns a PIL image with red overlay according to heat_01.
    """
    H, W = heat_01.shape
    base_resized = base_img.resize((W, H), Image.BICUBIC).convert("RGB")
    base_t = pil_to_tensor_no_numpy(base_resized).to(torch.float32)  # [3,H,W]

    red = (heat_01 * 255.0).clamp(0, 255).to(torch.uint8)
    red3 = torch.stack([red, torch.zeros_like(red), torch.zeros_like(red)], dim=0)  # [3,H,W]

    blended = ((1.0 - alpha) * base_t + alpha * red3.to(torch.float32)).clamp(0, 255).to(torch.uint8)
    raw = tensor_rgb_to_pil_bytes(blended)
    out_img = Image.frombytes("RGB", (W, H), raw)
    return out_img


def compute_per_layer_maps_clip(model: LeWrapper, image: torch.Tensor, text_embedding: torch.Tensor, patch_size: int) -> List[torch.Tensor]:
    """
    Return a list of per-layer heatmaps in [0,1], each [H, W] resized to the model input resolution.
    """
    assert text_embedding.ndim == 2 and text_embedding.shape[0] == 1
    _ = model.encode_image(image)

    blocks_list = list(dict(model.visual.transformer.resblocks.named_children()).values())
    image_features_list = []
    for layer in range(model.starting_depth, len(model.visual.transformer.resblocks)):
        intermediate_feat = model.visual.transformer.resblocks[layer].feat_post_mlp  # [num_patch, batch, dim]
        intermediate_feat = model.visual.ln_post(intermediate_feat.mean(dim=0)) @ model.visual.proj
        intermediate_feat = F.normalize(intermediate_feat, dim=-1)
        image_features_list.append(intermediate_feat)

    num_tokens = blocks_list[-1].feat_post_mlp.shape[0] - 1
    w = h = int(num_tokens ** 0.5)

    per_layer_maps = []
    for layer, (blk, img_feat) in enumerate(zip(blocks_list[model.starting_depth:], image_features_list)):
        model.visual.zero_grad()
        sim = text_embedding @ img_feat.transpose(-1, -2)  # [1, N]
        one_hot = torch.sum(sim)

        attn_map = blocks_list[model.starting_depth + layer].attn.attention_maps  # [(b*h), N, N]
        grad = torch.autograd.grad(one_hot, [attn_map], retain_graph=True, create_graph=True)[0]
        grad = rearrange(grad, '(b h) n m -> b h n m', b=1)  # [1,h,n,m]
        grad = torch.clamp(grad, min=0.)

        image_relevance = grad.mean(dim=1).mean(dim=1)[:, 1:]  # [1, N-1]
        expl_map = rearrange(image_relevance, 'b (ww hh) -> b ww hh', ww=w, hh=h)  # [1,w,h]
        expl_map = F.interpolate(expl_map.unsqueeze(1), scale_factor=patch_size, mode='bilinear').squeeze(1).squeeze(0)  # [H,W]
        # Min-max per layer
        mmin = float(expl_map.min())
        mmax = float(expl_map.max())
        expl_map = (expl_map - mmin) / (mmax - mmin + 1e-6)
        per_layer_maps.append(expl_map.detach().cpu())
    return per_layer_maps


def make_image_grid(images: List[Image.Image], cols: int) -> Image.Image:
    assert len(images) > 0
    w, h = images[0].size
    rows = ceil(len(images) / cols)
    grid = Image.new('RGB', (cols * w, rows * h))
    for idx, im in enumerate(images):
        r = idx // cols
        c = idx % cols
        grid.paste(im, (c * w, r * h))
    return grid


def plot_layer_activations(layers: List[int], vals_a: List[float], vals_b: List[float], label_a: str, label_b: str, title: str, out_path: str):
    plt.figure(figsize=(10, 4))
    plt.plot(layers, vals_a, label=label_a, marker='o')
    plt.plot(layers, vals_b, label=label_b, marker='o')
    plt.xlabel('Layer index')
    plt.ylabel('Mean LeGrad activation')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_layer_head_heatmap(matrix: torch.Tensor, title: str, out_path: str):
    """
    matrix: [L, H] tensor
    """
    plt.figure(figsize=(10, 6))
    plt.imshow(matrix.cpu().numpy(), aspect='auto', origin='lower', cmap='viridis')
    plt.colorbar(label='Mean LeGrad activation')
    plt.xlabel('Head index')
    plt.ylabel('Layer index')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def save_csv(layers: List[int], vals_a: List[float], vals_b: List[float], header_a: str, header_b: str, out_csv: str):
    with open(out_csv, 'w') as f:
        f.write('layer,{},{}\n'.format(header_a, header_b))
        for i, (va, vb) in enumerate(zip(vals_a, vals_b)):
            f.write(f"{layers[i]},{va:.6f},{vb:.6f}\n")


def save_head_csv(matrix: torch.Tensor, out_csv: str):
    """
    matrix: [L, H]
    """
    L, H = matrix.shape
    with open(out_csv, 'w') as f:
        headers = ['layer'] + [f'head_{h}' for h in range(H)]
        f.write(','.join(headers) + '\n')
        for l in range(L):
            vals = [f"{float(v):.6f}" for v in matrix[l]]
            f.write(','.join([str(l)] + vals) + '\n')


def main():
    parser = argparse.ArgumentParser(description='Analyze per-layer LeGrad activations for Cat/Dog dataset.')
    parser.add_argument('--dataset_root', type=str, default='/kaggle/input/dog-and-cat-classification-dataset/PetImages', help='Root with Cat/ and Dog/')
    parser.add_argument('--num_per_class', type=int, default=25, help='Images per class to sample')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--image_size', type=int, default=448)
    parser.add_argument('--model_name', type=str, default='ViT-B-16')
    parser.add_argument('--pretrained', type=str, default='laion2b_s34b_b88k')
    parser.add_argument('--output_dir', type=str, default='outputs')
    parser.add_argument('--prompt_a', type=str, default='a photo of a bird.', help='First prompt (scenario A)')
    parser.add_argument('--prompt_b', type=str, default='a photo of a human.', help='Second prompt (scenario B)')
    parser.add_argument('--prompts', type=str, nargs='*',
                        default=['a photo of a dog.', 'a photo of a cat.', 'a photo of a bird.', 'a photo of a human.'],
                        help='List of prompts to evaluate (e.g., 4 prompts: dog, cat, bird, human)')
    parser.add_argument('--class_a_subdir', type=str, default='Cat', help='Subfolder for class A images (e.g., cavallo)')
    parser.add_argument('--class_b_subdir', type=str, default='Dog', help='Subfolder for class B images (e.g., elefante)')
    parser.add_argument('--match_keyword_a', type=str, default='cat', help='Keyword to identify matching prompt for class A (e.g., horse)')
    parser.add_argument('--match_keyword_b', type=str, default='dog', help='Keyword to identify matching prompt for class B (e.g., elephant)')
    parser.add_argument('--save_layer_grids', type=int, default=0, help='If >0, save per-image per-layer overlay grids for this many images per class')
    parser.add_argument('--grid_cols', type=int, default=0, help='If 0, auto to (num_layers+1) for single-row. Otherwise, fixed columns.')
    parser.add_argument('--overlay_alpha', type=float, default=0.6, help='Overlay alpha for heatmaps')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, _, _ = open_clip.create_model_and_transforms(model_name=args.model_name, pretrained=args.pretrained, device=device)
    tokenizer = open_clip.get_tokenizer(model_name=args.model_name)
    model.eval()

    # Wrap with LeGrad, include all layers (layer_index=0)
    model = LeWrapper(model, layer_index=0)

    # Prepare text embeddings for prompts
    prompts = args.prompts if (args.prompts and len(args.prompts) > 0) else [args.prompt_a, args.prompt_b]
    tok = tokenizer(prompts).to(device)
    text_emb_all = model.encode_text(tok, normalize=True)  # [P, dim]

    class_a_dir = os.path.join(args.dataset_root, args.class_a_subdir)
    class_b_dir = os.path.join(args.dataset_root, args.class_b_subdir)
    class_a_paths = list_images(class_a_dir, limit=args.num_per_class, seed=args.seed)
    class_b_paths = list_images(class_b_dir, limit=args.num_per_class, seed=args.seed)

    # Utilities to accumulate per-layer activations for multiple prompts
    def process_batch_multi(image_paths: List[str], prompt_indices: List[int]):
        acc_layers = {pi: None for pi in prompt_indices}
        acc_heads = {pi: None for pi in prompt_indices}
        for pth in image_paths:
            try:
                img = Image.open(pth).convert('RGB')
            except Exception:
                continue
            img_t = safe_preprocess(img, image_size=args.image_size).unsqueeze(0).to(device)
            for pi in prompt_indices:
                per_layer = compute_per_layer_activation_clip(model, img_t, text_emb_all[pi:pi+1])
                per_heads = compute_per_layer_head_activation_clip(model, img_t, text_emb_all[pi:pi+1])  # [L,H]
                acc_layers[pi] = per_layer if acc_layers[pi] is None else acc_layers[pi] + per_layer
                acc_heads[pi] = per_heads if acc_heads[pi] is None else acc_heads[pi] + per_heads
        count = max(1, len(image_paths))
        acc_layers = {pi: acc_layers[pi] / count for pi in prompt_indices}
        acc_heads = {pi: acc_heads[pi] / count for pi in prompt_indices}
        return acc_layers, acc_heads

    prompt_indices = list(range(len(prompts)))
    # Class A images: process all prompts
    class_a_layers_dict, class_a_heads_dict = process_batch_multi(class_a_paths, prompt_indices)
    # Class B images: process all prompts
    class_b_layers_dict, class_b_heads_dict = process_batch_multi(class_b_paths, prompt_indices)

    # Determine match prompt per class via keyword heuristic
    def find_match_index(key: str) -> int:
        key = key.lower()
        for i, p in enumerate(prompts):
            if key in p.lower():
                return i
        return -1
    class_a_match_idx = find_match_index(args.match_keyword_a)
    class_b_match_idx = find_match_index(args.match_keyword_b)

    # Use any prompt to infer number of layers
    any_pi = prompt_indices[0]
    num_layers = int(class_a_layers_dict[any_pi].numel())
    layers = list(range(num_layers))

    # Single-line plot helper
    def plot_single_layer(layers_list: List[int], values: List[float], label: str, title: str, out_path: str):
        plt.figure(figsize=(10, 4))
        plt.plot(layers_list, values, label=label, marker='o')
        plt.xlabel('Layer index')
        plt.ylabel('Mean LeGrad activation')
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()

    san_class_a = sanitize(args.class_a_subdir)
    san_class_b = sanitize(args.class_b_subdir)

    # Save per-prompt outputs for Class A
    for pi in prompt_indices:
        pslug = sanitize(prompts[pi])
        vals = [float(v) for v in class_a_layers_dict[pi]]
        plot_single_layer(layers, vals, label=f'{prompts[pi]} on {args.class_a_subdir} images',
                          title=f'Per-layer LeGrad activation on {args.class_a_subdir} images',
                          out_path=os.path.join(args.output_dir, f'layers_{san_class_a}_images_{pslug}.png'))
        save_csv(layers, vals, vals, header_a=f'{pslug}_on_{san_class_a}', header_b=f'{pslug}_on_{san_class_a}',
                 out_csv=os.path.join(args.output_dir, f'layers_{san_class_a}_images_{pslug}.csv'))
        plot_layer_head_heatmap(class_a_heads_dict[pi], f'Heads: {prompts[pi]} on {args.class_a_subdir} images',
                                os.path.join(args.output_dir, f'heads_{san_class_a}_images_{pslug}.png'))
        save_head_csv(class_a_heads_dict[pi], os.path.join(args.output_dir, f'heads_{san_class_a}_images_{pslug}.csv'))
    # Differences vs class A match
    if class_a_match_idx >= 0:
        mslug = sanitize(prompts[class_a_match_idx])
        for pi in prompt_indices:
            if pi == class_a_match_idx:
                continue
            pslug = sanitize(prompts[pi])
            plot_layer_head_heatmap(class_a_heads_dict[pi] - class_a_heads_dict[class_a_match_idx],
                                    f'Heads: ({prompts[pi]} - {prompts[class_a_match_idx]}) on {args.class_a_subdir} images',
                                    os.path.join(args.output_dir, f'heads_{san_class_a}_images_diff_{pslug}_minus_{mslug}.png'))
            save_head_csv(class_a_heads_dict[pi] - class_a_heads_dict[class_a_match_idx],
                          os.path.join(args.output_dir, f'heads_{san_class_a}_images_diff_{pslug}_minus_{mslug}.csv'))

    # Save per-prompt outputs for Class B
    for pi in prompt_indices:
        pslug = sanitize(prompts[pi])
        vals = [float(v) for v in class_b_layers_dict[pi]]
        plot_single_layer(layers, vals, label=f'{prompts[pi]} on {args.class_b_subdir} images',
                          title=f'Per-layer LeGrad activation on {args.class_b_subdir} images',
                          out_path=os.path.join(args.output_dir, f'layers_{san_class_b}_images_{pslug}.png'))
        save_csv(layers, vals, vals, header_a=f'{pslug}_on_{san_class_b}', header_b=f'{pslug}_on_{san_class_b}',
                 out_csv=os.path.join(args.output_dir, f'layers_{san_class_b}_images_{pslug}.csv'))
        plot_layer_head_heatmap(class_b_heads_dict[pi], f'Heads: {prompts[pi]} on {args.class_b_subdir} images',
                                os.path.join(args.output_dir, f'heads_{san_class_b}_images_{pslug}.png'))
        save_head_csv(class_b_heads_dict[pi], os.path.join(args.output_dir, f'heads_{san_class_b}_images_{pslug}.csv'))
    # Differences vs class B match
    if class_b_match_idx >= 0:
        mslug = sanitize(prompts[class_b_match_idx])
        for pi in prompt_indices:
            if pi == class_b_match_idx:
                continue
            pslug = sanitize(prompts[pi])
            plot_layer_head_heatmap(class_b_heads_dict[pi] - class_b_heads_dict[class_b_match_idx],
                                    f'Heads: ({prompts[pi]} - {prompts[class_b_match_idx]}) on {args.class_b_subdir} images',
                                    os.path.join(args.output_dir, f'heads_{san_class_b}_images_diff_{pslug}_minus_{mslug}.png'))
            save_head_csv(class_b_heads_dict[pi] - class_b_heads_dict[class_b_match_idx],
                          os.path.join(args.output_dir, f'heads_{san_class_b}_images_diff_{pslug}_minus_{mslug}.csv'))

    # Optional: save per-layer grids for a subset of images
    if args.save_layer_grids > 0:
        grids_dir = os.path.join(args.output_dir, 'layer_grids')
        os.makedirs(grids_dir, exist_ok=True)
        # Helper to save for a list of paths
        def save_grids_for_paths(paths: List[str], class_tag: str):
            for pth in paths[:args.save_layer_grids]:
                try:
                    base_img = Image.open(pth).convert('RGB')
                except Exception:
                    continue
                # Preprocess to get image size and fill features (we'll use base_img resized in overlay)
                img_tensor = safe_preprocess(base_img, image_size=args.image_size).unsqueeze(0).to(device)
                # For all prompts
                for idx_prompt, prompt in enumerate(prompts):
                    te = text_emb_all[idx_prompt:idx_prompt+1]
                    per_layer_maps = compute_per_layer_maps_clip(model, img_tensor , te, patch_size=model.patch_size)
                    # Build grid: original + per-layer overlays
                    overlays = [base_img.resize((per_layer_maps[0].shape[1], per_layer_maps[0].shape[0]), Image.BICUBIC)]
                    overlays[0] = overlays[0]  # original
                    for hm in per_layer_maps:
                        overlays.append(overlay_heat_to_pil(overlays[0], hm, alpha=args.overlay_alpha))
                    cols = args.grid_cols if args.grid_cols > 0 else len(overlays)
                    grid = make_image_grid(overlays, cols=cols)
                    base_name = os.path.splitext(os.path.basename(pth))[0]
                    out_name = f"{class_tag}_{base_name}_{sanitize(prompt)}_layers_grid.png"
                    grid.save(os.path.join(grids_dir, out_name))
        save_grids_for_paths(class_a_paths, san_class_a)
        save_grids_for_paths(class_b_paths, san_class_b)

    print('Saved plots and CSVs in:', args.output_dir)


if __name__ == '__main__':
    main()


