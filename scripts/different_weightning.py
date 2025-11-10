#!/usr/bin/env python
import argparse
import os
import random
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


def compute_standard_and_weighted_maps_clip(
	model: LeWrapper,
	image: torch.Tensor,
	text_embedding: torch.Tensor,
	focus_layer_index: int = 10,
	focus_head_index: int = 10,
) -> Tuple[torch.Tensor, torch.Tensor]:
	"""
	Returns:
	- standard_map: [1, P, H, W] standard LeGrad (average over layers and heads)
	- weighted_map: [1, P, H, W] custom weighting with 50% from (focus_layer_index, focus_head_index)
	"""
	assert text_embedding.ndim == 2
	num_prompts = text_embedding.shape[0]

	# Replicate image for prompts, populate hooks/features
	if image is not None:
		_ = model.encode_image(image)  # no repeat; keep batch=1 for lower memory

	blocks_list = list(dict(model.visual.transformer.resblocks.named_children()).values())

	image_features_list = []
	for layer in range(model.starting_depth, len(model.visual.transformer.resblocks)):
		intermediate_feat = model.visual.transformer.resblocks[layer].feat_post_mlp  # [num_patch, batch, dim]
		intermediate_feat = model.visual.ln_post(intermediate_feat.mean(dim=0)) @ model.visual.proj
		intermediate_feat = F.normalize(intermediate_feat, dim=-1)
		image_features_list.append(intermediate_feat)

	num_tokens = blocks_list[-1].feat_post_mlp.shape[0] - 1
	w = h = int(num_tokens ** 0.5)

	total_layers = len(blocks_list) - model.starting_depth
	# Clamp focus indices in range
	focus_layer_index = max(0, min(total_layers - 1, focus_layer_index))

	# We need head count to clamp focus head
	# Do a quick forward access to attn maps to know head count at the first layer
	first_attn = blocks_list[model.starting_depth].attn.attention_maps  # [(b*h), N, N]
	# since we kept batch=1, head count equals first_attn.shape[0]
	num_heads = first_attn.shape[0]
	focus_head_index = max(0, min(num_heads - 1, focus_head_index))

	# Accumulators
	sum_per_layer_mean = 0.0  # sum over layers of (mean over heads) -> shape [1, P, H, W]
	sum_all_pairs = 0.0       # sum over all layer-head pairs         -> shape [1, P, H, W]
	target_map = None         # map for focus (layer, head)           -> shape [1, P, H, W]

	for layer, (blk, img_feat) in enumerate(zip(blocks_list[model.starting_depth:], image_features_list)):
		model.visual.zero_grad()
		sim = text_embedding @ img_feat.transpose(-1, -2)  # [P, N]
		one_hot = torch.sum(sim)  # scalar objective across prompts

		attn_map = blocks_list[model.starting_depth + layer].attn.attention_maps  # [(b*h), N, N]
		grad = torch.autograd.grad(one_hot, [attn_map], retain_graph=True, create_graph=False)[0]
		# batch is 1 here
		grad = rearrange(grad, '(b h) n m -> b h n m', b=1)  # [1, H, N, N]
		grad = torch.clamp(grad, min=0.)

		# Average over queries, drop CLS on key dim
		head_token_relevance = grad.mean(dim=2)[:, :, 1:]  # [1, H, N-1]

		# Per-layer mean over heads
		layer_mean = head_token_relevance.mean(dim=1)  # [1, N-1]
		layer_mean_map = rearrange(layer_mean, 'b (ww hh) -> 1 b ww hh', ww=w, hh=h)  # [1, P, w, h]
		layer_mean_map = F.interpolate(layer_mean_map, scale_factor=model.patch_size, mode='bilinear')  # [1, P, H, W]
		sum_per_layer_mean = sum_per_layer_mean + layer_mean_map

		# Sum over heads equals H * (mean over heads)
		layer_sum_heads = layer_mean_map * float(num_heads)  # [1, P, H, W]
		sum_all_pairs = sum_all_pairs + layer_sum_heads

		# If this is the focus layer, extract the focus head map
		if layer == focus_layer_index:
			head_focus = head_token_relevance[:, focus_head_index, :]  # [1, N-1]
			head_focus_map = rearrange(head_focus, 'b (ww hh) -> 1 b ww hh', ww=w, hh=h)
			head_focus_map = F.interpolate(head_focus_map, scale_factor=model.patch_size, mode='bilinear')  # [1, P, H, W]
			target_map = head_focus_map

	# Standard LeGrad: average over layers then min-max
	standard_map = min_max_batch(sum_per_layer_mean)

	# Weighted: 50% from focus (layer, head), 50% uniformly from the rest of pairs
	if target_map is None:
		# Fallback: if something went wrong, just duplicate standard
		weighted_map = standard_map.clone()
	else:
		num_pairs = float(total_layers * num_heads)
		rest_mean = (sum_all_pairs - target_map) / max(1.0, (num_pairs - 1.0))
		weighted = 0.5 * target_map + 0.5 * rest_mean
		weighted_map = min_max_batch(weighted)

	return standard_map, weighted_map


def overlay(ax, base_img: Image.Image, heat_01: torch.Tensor, title: str, alpha: float = 0.6):
	# heat_01: [H, W] float in [0, 1]
	H, W = heat_01.shape
	base_resized = base_img.resize((W, H), Image.BICUBIC).convert("RGB")
	ax.imshow(base_resized)
	ax.imshow(heat_01.detach().cpu().numpy(), cmap='jet', alpha=alpha, vmin=0.0, vmax=1.0)
	ax.set_title(title, fontsize=10)
	ax.axis('off')


def main():
	parser = argparse.ArgumentParser(description='Compare standard LeGrad vs weighted (50% layer-10 head-10).')
	parser.add_argument('--dataset_root', type=str, required=True)
	parser.add_argument('--num_per_class', type=int, default=5)
	parser.add_argument('--image_size', type=int, default=448)
	parser.add_argument('--model_name', type=str, default='ViT-B-16')
	parser.add_argument('--pretrained', type=str, default='laion2b_s34b_b88k')
	parser.add_argument('--prompts', type=str, nargs='*',
	                    default=['a photo of a dog.', 'a photo of a cat.', 'a photo of a bird.', 'a photo of a human.'])
	parser.add_argument('--output_path', type=str, default='outputs/different_weighting_comparison.png')
	parser.add_argument('--seed', type=int, default=42)
	args = parser.parse_args()

	os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	model, _, _ = open_clip.create_model_and_transforms(model_name=args.model_name, pretrained=args.pretrained, device=device)
	tokenizer = open_clip.get_tokenizer(model_name=args.model_name)
	model.eval()
	# Wrap to enable LeGrad hooks and multi-layer access
	model = LeWrapper(model, layer_index=0)

	# Text embeddings
	tok = tokenizer(args.prompts).to(device)
	text_emb = model.encode_text(tok, normalize=True)  # [P, dim]

	# Collect images: try Cat and Dog subfolders; fallback to flat listing
	paths = []
	cat_dir = os.path.join(args.dataset_root, 'Cat')
	dog_dir = os.path.join(args.dataset_root, 'Dog')
	if os.path.isdir(cat_dir) and os.path.isdir(dog_dir):
		paths += list_images(cat_dir, limit=args.num_per_class, seed=args.seed)
		paths += list_images(dog_dir, limit=args.num_per_class, seed=args.seed)
	else:
		paths += list_images(args.dataset_root, limit=max(1, args.num_per_class * 2), seed=args.seed)

	if len(paths) == 0:
		raise RuntimeError(f'No images found under {args.dataset_root}')

	# Accumulate average maps across images
	sum_std = None
	sum_wtd = None
	base_image_for_overlay = None
	count = 0
	for pth in paths:
		try:
			img = Image.open(pth).convert('RGB')
		except Exception:
			continue
		if base_image_for_overlay is None:
			base_image_for_overlay = img.copy()
		img_t = safe_preprocess(img, image_size=args.image_size).unsqueeze(0).to(device)

		with torch.enable_grad():
			std_map, wtd_map = compute_standard_and_weighted_maps_clip(model, img_t, text_emb,
			                                                           focus_layer_index=10, focus_head_index=10)
		sum_std = std_map if sum_std is None else (sum_std + std_map)
		sum_wtd = wtd_map if sum_wtd is None else (sum_wtd + wtd_map)
		count += 1

	# Mean over images
	mean_std = sum_std / max(1, count)     # [1, P, H, W]
	mean_wtd = sum_wtd / max(1, count)     # [1, P, H, W]

	# Build a single figure: rows = prompts, cols = 2 (standard vs weighted)
	P = len(args.prompts)
	fig, axes = plt.subplots(nrows=P, ncols=2, figsize=(8, 3 * P))
	if P == 1:
		axes = [axes]  # normalize to list of rows
	for i in range(P):
		std_i = mean_std[0, i]
		wtd_i = mean_wtd[0, i]
		overlay(axes[i][0], base_image_for_overlay, std_i, title=f'{args.prompts[i]} - LeGrad', alpha=0.6)
		overlay(axes[i][1], base_image_for_overlay, wtd_i, title=f'{args.prompts[i]} - Weighted (50% L10-H10)', alpha=0.6)
	plt.tight_layout()
	plt.savefig(args.output_path, dpi=150)
	plt.close(fig)

	print(f'Saved comparison figure to: {args.output_path}')


if __name__ == '__main__':
	main()


