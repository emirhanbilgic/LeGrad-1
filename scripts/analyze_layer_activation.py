#!/usr/bin/env python

import argparse
import os
import random
from io import BytesIO
from typing import List, Tuple
import re

import requests
from PIL import Image

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
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, _, _ = open_clip.create_model_and_transforms(model_name=args.model_name, pretrained=args.pretrained, device=device)
    tokenizer = open_clip.get_tokenizer(model_name=args.model_name)
    model.eval()

    # Wrap with LeGrad, include all layers (layer_index=0)
    model = LeWrapper(model, layer_index=0)

    # Prepare text embeddings for two prompts
    prompts = [args.prompt_a, args.prompt_b]
    tok = tokenizer(prompts).to(device)
    text_emb_all = model.encode_text(tok, normalize=True)  # [2, dim]

    cat_dir = os.path.join(args.dataset_root, 'Cat')
    dog_dir = os.path.join(args.dataset_root, 'Dog')
    cat_paths = list_images(cat_dir, limit=args.num_per_class, seed=args.seed)
    dog_paths = list_images(dog_dir, limit=args.num_per_class, seed=args.seed)

    # Utilities to accumulate per-layer activations
    def process_batch(image_paths: List[str], prompt_idx_a: int, prompt_idx_b: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        acc_a, acc_b = None, None
        acc_heads_a, acc_heads_b = None, None
        for pth in image_paths:
            try:
                img = Image.open(pth).convert('RGB')
            except Exception:
                continue
            img_t = safe_preprocess(img, image_size=args.image_size).unsqueeze(0).to(device)
            per_layer_a = compute_per_layer_activation_clip(model, img_t, text_emb_all[prompt_idx_a:prompt_idx_a+1])
            per_layer_b = compute_per_layer_activation_clip(model, img_t, text_emb_all[prompt_idx_b:prompt_idx_b+1])
            per_heads_a = compute_per_layer_head_activation_clip(model, img_t, text_emb_all[prompt_idx_a:prompt_idx_a+1])  # [L,H]
            per_heads_b = compute_per_layer_head_activation_clip(model, img_t, text_emb_all[prompt_idx_b:prompt_idx_b+1])  # [L,H]
            acc_a = per_layer_a if acc_a is None else acc_a + per_layer_a
            acc_b = per_layer_b if acc_b is None else acc_b + per_layer_b
            acc_heads_a = per_heads_a if acc_heads_a is None else acc_heads_a + per_heads_a
            acc_heads_b = per_heads_b if acc_heads_b is None else acc_heads_b + per_heads_b
        count = max(1, len(image_paths))
        return acc_a / count, acc_b / count, acc_heads_a / count, acc_heads_b / count

    # Cat images: matched(cat) vs mismatched(dog)
    cat_match, cat_mismatch, cat_heads_match, cat_heads_mismatch = process_batch(cat_paths, prompt_idx_a=0, prompt_idx_b=1)
    # Dog images: matched(dog) vs mismatched(cat)
    dog_match, dog_mismatch, dog_heads_match, dog_heads_mismatch = process_batch(dog_paths, prompt_idx_a=1, prompt_idx_b=0)

    num_layers = int(cat_match.numel())
    layers = list(range(num_layers))

    san_pa = sanitize(args.prompt_a)
    san_pb = sanitize(args.prompt_b)

    # Convert to python lists
    cat_match_l = [float(v) for v in cat_match]
    cat_mismatch_l = [float(v) for v in cat_mismatch]
    dog_match_l = [float(v) for v in dog_match]
    dog_mismatch_l = [float(v) for v in dog_mismatch]

    # Plots
    plot_layer_activations(layers, cat_match_l, cat_mismatch_l,
                           label_a=f'{args.prompt_a} on cat images',
                           label_b=f'{args.prompt_b} on cat images',
                           title='Per-layer LeGrad activation on Cat images',
                           out_path=os.path.join(args.output_dir, f'layers_cat_images_{san_pa}_vs_{san_pb}.png'))

    plot_layer_activations(layers, dog_match_l, dog_mismatch_l,
                           label_a=f'{args.prompt_a} on dog images',
                           label_b=f'{args.prompt_b} on dog images',
                           title='Per-layer LeGrad activation on Dog images',
                           out_path=os.path.join(args.output_dir, f'layers_dog_images_{san_pa}_vs_{san_pb}.png'))

    # CSVs
    save_csv(layers, cat_match_l, cat_mismatch_l,
             header_a=f'{san_pa}_on_cat', header_b=f'{san_pb}_on_cat',
             out_csv=os.path.join(args.output_dir, f'layers_cat_images_{san_pa}_vs_{san_pb}.csv'))
    save_csv(layers, dog_match_l, dog_mismatch_l,
             header_a=f'{san_pa}_on_dog', header_b=f'{san_pb}_on_dog',
             out_csv=os.path.join(args.output_dir, f'layers_dog_images_{san_pa}_vs_{san_pb}.csv'))

    # Per-head heatmaps and CSVs
    plot_layer_head_heatmap(cat_heads_match, f'Heads: {args.prompt_a} on cat images',
                            os.path.join(args.output_dir, f'heads_cat_images_{san_pa}.png'))
    plot_layer_head_heatmap(cat_heads_mismatch, f'Heads: {args.prompt_b} on cat images',
                            os.path.join(args.output_dir, f'heads_cat_images_{san_pb}.png'))
    plot_layer_head_heatmap(cat_heads_mismatch - cat_heads_match, f'Heads: ({args.prompt_b} - {args.prompt_a}) on cat images',
                            os.path.join(args.output_dir, f'heads_cat_images_diff_{san_pb}_minus_{san_pa}.png'))
    save_head_csv(cat_heads_match, os.path.join(args.output_dir, f'heads_cat_images_{san_pa}.csv'))
    save_head_csv(cat_heads_mismatch, os.path.join(args.output_dir, f'heads_cat_images_{san_pb}.csv'))
    save_head_csv(cat_heads_mismatch - cat_heads_match, os.path.join(args.output_dir, f'heads_cat_images_diff_{san_pb}_minus_{san_pa}.csv'))

    plot_layer_head_heatmap(dog_heads_match, f'Heads: {args.prompt_a} on dog images',
                            os.path.join(args.output_dir, f'heads_dog_images_{san_pa}.png'))
    plot_layer_head_heatmap(dog_heads_mismatch, f'Heads: {args.prompt_b} on dog images',
                            os.path.join(args.output_dir, f'heads_dog_images_{san_pb}.png'))
    plot_layer_head_heatmap(dog_heads_mismatch - dog_heads_match, f'Heads: ({args.prompt_b} - {args.prompt_a}) on dog images',
                            os.path.join(args.output_dir, f'heads_dog_images_diff_{san_pb}_minus_{san_pa}.png'))
    save_head_csv(dog_heads_match, os.path.join(args.output_dir, f'heads_dog_images_{san_pa}.csv'))
    save_head_csv(dog_heads_mismatch, os.path.join(args.output_dir, f'heads_dog_images_{san_pb}.csv'))
    save_head_csv(dog_heads_mismatch - dog_heads_match, os.path.join(args.output_dir, f'heads_dog_images_diff_{san_pb}_minus_{san_pa}.csv'))

    print('Saved plots and CSVs in:', args.output_dir)


if __name__ == '__main__':
    main()


