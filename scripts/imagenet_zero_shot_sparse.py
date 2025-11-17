#!/usr/bin/env python
"""
Zero-shot ImageNet evaluation for CLIP with and without sparse text encodings.

This script:
- Builds prompts for all 1000 ImageNet classes (e.g. "a photo of a baseball player.")
- Forms a dictionary D of all class text embeddings.
- For each class, computes a sparse residual encoding using OMP with a fixed
  number of atoms (default: 8), excluding the class itself from its dictionary.
- Evaluates zero-shot top-1 and top-5 accuracy on ImageNet using:
    (1) standard CLIP text embeddings
    (2) sparse-residual text embeddings

The ImageNet data is expected as an ImageFolder under:
    <imagenet_root>/<split>/
with 1000 subdirectories named by WordNet IDs (e.g. n01440764, ...),
which is the standard layout after unpacking ILSVRC2012.
"""

import argparse
import json
import os
from typing import Dict, List, Tuple

import requests
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import open_clip


IMAGENET_CLASS_INDEX_URL = (
    "https://raw.githubusercontent.com/pytorch/vision/main/torchvision/datasets/imagenet_class_index.json"
)


def download_imagenet_class_index(dst_path: str) -> Dict[str, Tuple[str, str]]:
    """
    Download the standard ImageNet class index JSON used by torchvision.

    Returns a dict mapping string class ids "0".."999" to (wnid, human_readable_label).
    The file is also cached at dst_path.
    """
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    resp = requests.get(IMAGENET_CLASS_INDEX_URL, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    with open(dst_path, "w") as f:
        json.dump(data, f)
    return data


def load_imagenet_class_index(path: str) -> Dict[str, Tuple[str, str]]:
    """
    Load (or download) the ImageNet class index JSON.
    """
    if os.path.isfile(path):
        with open(path, "r") as f:
            data = json.load(f)
        return data
    # Fallback: try to download
    print(f"[imagenet] class index not found at {path}, downloading from {IMAGENET_CLASS_INDEX_URL} ...")
    return download_imagenet_class_index(path)


def build_wnid_to_label_map(class_index: Dict[str, List[str]]) -> Dict[str, str]:
    """
    Convert torchvision's ImageNet class index mapping into wnid -> primary label.

    Each entry in class_index is:
        idx_str -> [wnid, "label or comma-separated labels"]
    We take the first comma-separated segment as the primary label and lowercase it.
    """
    wnid_to_label: Dict[str, str] = {}
    for _, (wnid, label_str) in class_index.items():
        # Use the primary label (before first comma) and replace underscores with spaces
        # so prompts use natural phrases like "great white shark" instead of "great_white_shark".
        primary = label_str.split(",")[0].replace("_", " ").strip()
        wnid_to_label[wnid] = primary.lower()
    return wnid_to_label


def build_prompts_for_imagenet_classes(
    wnids: List[str],
    wnid_to_label: Dict[str, str],
    template: str = "a photo of a {}.",
) -> List[str]:
    """
    Build natural language prompts for each ImageNet class.

    - wnids: list of WordNet IDs in the order used by the dataset.
    - wnid_to_label: maps wnid -> human-readable label (possibly multi-word).
    - template: e.g. "a photo of a {}." or "a photo of the {}."

    If a wnid is missing in wnid_to_label, we fall back to using the wnid itself.
    """
    prompts: List[str] = []
    for wnid in wnids:
        label = wnid_to_label.get(wnid, wnid)
        prompt = template.format(label)
        prompts.append(prompt)
    return prompts


def omp_sparse_residual(x_1x: torch.Tensor, D: torch.Tensor, max_atoms: int = 8, tol: float = 1e-6) -> torch.Tensor:
    """
    Orthogonal Matching Pursuit residual.

    Args:
        x_1x: [1, d] L2-normalized text embedding for a single class.
        D:    [K, d] dictionary atoms (L2-normalized), typically all other class embeddings.
        max_atoms: maximum number of atoms to select.
        tol: tolerance for early stopping.

    Returns:
        r: [1, d] L2-normalized residual (or x_1x if dictionary is degenerate).
    """
    if D is None or D.numel() == 0:
        return F.normalize(x_1x, dim=-1)
    x = x_1x.clone()  # [1, d]
    K = D.shape[0]
    max_atoms = int(max(1, min(max_atoms, K)))
    selected: List[int] = []
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
        D_S = D[selected, :]  # [t, d]
        G = D_S @ D_S.t()     # [t, t]
        b = (D_S @ x.t())     # [t, 1]
        I = torch.eye(G.shape[0], device=G.device, dtype=G.dtype)
        s = torch.linalg.solve(G + 1e-6 * I, b)  # [t, 1]
        x_hat = (s.t() @ D_S).to(x.dtype)        # [1, d]
        r = (x - x_hat)
        if float(torch.norm(r)) <= tol:
            break
    if torch.norm(r) <= tol:
        return F.normalize(x, dim=-1)
    return F.normalize(r, dim=-1)


@torch.no_grad()
def compute_text_embeddings(
    model,
    tokenizer,
    prompts: List[str],
    device: torch.device,
) -> torch.Tensor:
    """
    Encode a list of prompts with CLIP and L2-normalize.
    """
    batch_size = 256  # encode in chunks to avoid OOM
    all_embs: List[torch.Tensor] = []
    for i in range(0, len(prompts), batch_size):
        chunk = prompts[i : i + batch_size]
        tokens = tokenizer(chunk).to(device)
        emb = model.encode_text(tokens)
        emb = F.normalize(emb, dim=-1)
        all_embs.append(emb)
    return torch.cat(all_embs, dim=0)


def precompute_sparse_text_embeddings(
    text_embs: torch.Tensor,
    atoms: int,
    max_cos_sim: float | None = 0.9,
) -> torch.Tensor:
    """
    For each class embedding, compute the sparse residual encoding using
    dictionary of all OTHER class embeddings (no self-atom).

    Args:
        text_embs: [C, d] L2-normalized embeddings for C classes.
        atoms: maximum atoms per OMP run.

    Returns:
        sparse_embs: [C, d] L2-normalized residual embeddings.
    """
    C, d = text_embs.shape
    device = text_embs.device
    sparse_embs = torch.empty_like(text_embs)
    print(f"[sparse] Precomputing sparse encodings for {C} classes with {atoms} atoms ...")
    for c in tqdm(range(C), desc="OMP (classes)"):
        x = text_embs[c : c + 1]  # [1, d]
        if C == 1:
            sparse_embs[c] = x
            continue
        # dictionary: all other classes
        parts = []
        if c > 0:
            parts.append(text_embs[:c])
        if c + 1 < C:
            parts.append(text_embs[c + 1 :])
        D = torch.cat(parts, dim=0) if len(parts) > 0 else text_embs.new_zeros((0, d))
        if D.numel() > 0:
            D = F.normalize(D, dim=-1)
            # Optionally remove atoms that are almost identical to x (too high cosine similarity)
            if max_cos_sim is not None and float(max_cos_sim) < 1.0:
                sim = (D @ x.t()).squeeze(-1).abs()  # [K]
                keep = sim < float(max_cos_sim)
                D = D[keep]
        # If the dictionary becomes empty after filtering, fall back to original embedding
        if D.numel() == 0:
            sparse_embs[c] = x
            continue
        sparse = omp_sparse_residual(x, D, max_atoms=atoms)
        sparse_embs[c] = sparse.to(device)
    return sparse_embs


@torch.no_grad()
def evaluate_imagenet(
    model,
    dataloader: DataLoader,
    text_embs: torch.Tensor,
    sparse_text_embs: torch.Tensor,
    device: torch.device,
) -> None:
    """
    Run zero-shot evaluation and report top-1 / top-5 accuracy for:
      - standard CLIP text embeddings
      - sparse residual text embeddings
    """
    model.eval()
    text_embs = text_embs.to(device)
    sparse_text_embs = sparse_text_embs.to(device)

    correct_top1_std = 0
    correct_top5_std = 0
    correct_top1_sparse = 0
    correct_top5_sparse = 0
    total = 0

    for images, targets in tqdm(dataloader, desc="Evaluating", leave=False):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # Encode images
        image_features = model.encode_image(images)
        image_features = F.normalize(image_features, dim=-1)

        # Standard logits
        logits_std = image_features @ text_embs.t()
        # Sparse-encoded logits
        logits_sparse = image_features @ sparse_text_embs.t()

        # Optionally apply logit_scale (does not affect argmax but affects calibration)
        if hasattr(model, "logit_scale"):
            logit_scale = model.logit_scale.exp()
            logits_std = logit_scale * logits_std
            logits_sparse = logit_scale * logits_sparse

        # Top-1 / Top-5 for standard
        _, pred_std_top5 = logits_std.topk(5, dim=-1)
        correct_std_top1_batch = (pred_std_top5[:, 0] == targets).sum().item()
        correct_std_top5_batch = (pred_std_top5 == targets.unsqueeze(1)).any(dim=1).sum().item()

        # Top-1 / Top-5 for sparse
        _, pred_sparse_top5 = logits_sparse.topk(5, dim=-1)
        correct_sparse_top1_batch = (pred_sparse_top5[:, 0] == targets).sum().item()
        correct_sparse_top5_batch = (pred_sparse_top5 == targets.unsqueeze(1)).any(dim=1).sum().item()

        batch_size = targets.size(0)
        total += batch_size
        correct_top1_std += correct_std_top1_batch
        correct_top5_std += correct_std_top5_batch
        correct_top1_sparse += correct_sparse_top1_batch
        correct_top5_sparse += correct_sparse_top5_batch

    top1_std = 100.0 * correct_top1_std / total
    top5_std = 100.0 * correct_top5_std / total
    top1_sparse = 100.0 * correct_top1_sparse / total
    top5_sparse = 100.0 * correct_top5_sparse / total

    print("=== Zero-shot ImageNet results ===")
    print(f"Total images: {total}")
    print(f"[Standard CLIP]  Top-1: {top1_std:.2f}%   Top-5: {top5_std:.2f}%")
    print(f"[Sparse (atoms=8)] Top-1: {top1_sparse:.2f}%   Top-5: {top5_sparse:.2f}%")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Zero-shot CLIP on ImageNet with sparse text encodings (OMP residual)."
    )
    parser.add_argument(
        "--imagenet_root",
        type=str,
        required=True,
        help="Root directory of ImageNet (expects <root>/<split>/ with 1000 class folders).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["train", "val"],
        help="Which split to evaluate (default: val).",
    )
    parser.add_argument(
        "--class_index_path",
        type=str,
        default="resources/imagenet_class_index.json",
        help="Path to imagenet_class_index.json (downloaded automatically if missing).",
    )
    parser.add_argument(
        "--prompt_template",
        type=str,
        default="a photo of a {}.",
        help="Prompt template; {} will be replaced by the full ImageNet class label (may contain multiple words).",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="ViT-B-16",
        help="OpenCLIP model name (e.g., ViT-B-16).",
    )
    parser.add_argument(
        "--pretrained",
        type=str,
        default="laion2b_s34b_b88k",
        help="OpenCLIP pretrained tag.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for evaluation.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of dataloader workers.",
    )
    parser.add_argument(
        "--atoms",
        type=int,
        default=8,
        help="Number of atoms for OMP sparse residual (dictionary = other 999 class prompts).",
    )
    parser.add_argument(
        "--max_dict_cos_sim",
        type=float,
        default=0.9,
        help="Maximum allowed cosine similarity between a class embedding and dictionary atoms. "
             "Atoms with abs(cos) >= this are removed from the dictionary before OMP. "
             "Set >=1.0 or <=0 to disable.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (cuda or cpu).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cuda")
    print(f"[device] Using device: {device}")

    # 1) Load ImageNet dataset as ImageFolder
    split_dir = os.path.join(args.imagenet_root, args.split)
    if not os.path.isdir(split_dir):
        raise RuntimeError(
            f"Expected ImageNet {args.split} directory at {split_dir} with 1000 class subfolders."
        )

    # Create CLIP model & preprocessing
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name=args.model_name,
        pretrained=args.pretrained,
        device=device,
    )
    tokenizer = open_clip.get_tokenizer(args.model_name)
    model.eval()

    # ImageNet dataset
    dataset = datasets.ImageFolder(root=split_dir, transform=preprocess)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    print(f"[imagenet] Loaded split '{args.split}' from {split_dir} with {len(dataset)} images.")

    # 2) Build prompts for all classes using official ImageNet labels (may be multi-word)
    class_index = load_imagenet_class_index(args.class_index_path)
    wnid_to_label = build_wnid_to_label_map(class_index)

    wnids: List[str] = list(dataset.classes)  # e.g. ["n01440764", ...] in the dataset's class index order
    prompts = build_prompts_for_imagenet_classes(
        wnids=wnids,
        wnid_to_label=wnid_to_label,
        template=args.prompt_template,
    )

    print(f"[prompts] Built {len(prompts)} prompts. Example:")
    for i in range(min(5, len(prompts))):
        print(f"  class {i:3d} ({wnids[i]}): {prompts[i]}")

    # 3) Compute standard CLIP text embeddings for all prompts
    text_embs = compute_text_embeddings(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        device=device,
    )

    # 4) Precompute sparse residual embeddings with dictionary of all other class prompts
    sparse_text_embs = precompute_sparse_text_embeddings(
        text_embs=text_embs,
        atoms=args.atoms,
        max_cos_sim=args.max_dict_cos_sim if 0.0 < args.max_dict_cos_sim < 1.0 else None,
    )

    # 5) Evaluate zero-shot accuracy
    evaluate_imagenet(
        model=model,
        dataloader=dataloader,
        text_embs=text_embs,
        sparse_text_embs=sparse_text_embs,
        device=device,
    )


if __name__ == "__main__":
    main()


