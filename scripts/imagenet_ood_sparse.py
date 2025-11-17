#!/usr/bin/env python
"""
OOD detection on ImageNet using CLIP with and without sparse text encodings.

Workflow:
 1) Build prompts for the 1000 ImageNet-1k classes (e.g. "a photo of a great white shark.").
 2) Encode all prompts with CLIP → text embeddings (dictionary).
 3) For each class, compute a sparse residual embedding via OMP using all OTHER classes
    as dictionary atoms (with optional cosine-similarity filtering).
 4) For ID data (ImageNet val) and OOD data (imagenetood), compute:
       - scores_std    = max_j logit_std(x, class_j)
       - scores_sparse = max_j logit_sparse(x, class_j)
    where logits are image_features @ text_features^T.
 5) Treat ID as positive, OOD as negative and compute OOD metrics:
       - AUROC
       - FPR@95%TPR
       - AUPR (ID positive), AUPR (OOD positive)

The OOD data is expected at:
    <ood_root>/
with images either directly under that folder or in arbitrary subfolders.
"""

import argparse
import os
import sys
from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from PIL import Image
from tqdm import tqdm

import open_clip


# Make helpers from imagenet_zero_shot_sparse.py importable
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(ROOT_DIR, "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.append(SCRIPTS_DIR)

import imagenet_zero_shot_sparse as izs  # type: ignore


class FlatImageDataset(Dataset):
    """
    Simple dataset that loads all images under a root (recursively),
    without using class labels (OOD set is unlabeled).
    """

    def __init__(self, root: str, transform):
        self.root = root
        self.transform = transform
        self.paths: List[str] = []
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        for r, _, files in os.walk(root):
            for name in files:
                if os.path.splitext(name.lower())[1] in exts:
                    self.paths.append(os.path.join(r, name))
        if len(self.paths) == 0:
            raise RuntimeError(f"No images found under {root}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        path = self.paths[idx]
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        # dummy label (unused)
        return img, 0


@torch.no_grad()
def collect_maxlogit_scores(
    model,
    dataloader: DataLoader,
    text_embs: torch.Tensor,
    sparse_text_embs: torch.Tensor,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    For each image, compute:
        score_std    = max_j logits_std
        score_sparse = max_j logits_sparse
    Returns two 1D tensors of shape [N].
    """
    model.eval()
    text_embs = text_embs.to(device)
    sparse_text_embs = sparse_text_embs.to(device)

    scores_std: List[torch.Tensor] = []
    scores_sparse: List[torch.Tensor] = []

    for images, _ in tqdm(dataloader, desc="Scoring", leave=False):
        images = images.to(device, non_blocking=True)
        image_features = model.encode_image(images)
        image_features = F.normalize(image_features, dim=-1)

        logits_std = image_features @ text_embs.t()
        logits_sparse = image_features @ sparse_text_embs.t()

        if hasattr(model, "logit_scale"):
            scale = model.logit_scale.exp()
            logits_std = scale * logits_std
            logits_sparse = scale * logits_sparse

        scores_std.append(logits_std.max(dim=-1).values.cpu())
        scores_sparse.append(logits_sparse.max(dim=-1).values.cpu())

    return torch.cat(scores_std, dim=0), torch.cat(scores_sparse, dim=0)


def _binary_auroc(pos_scores: torch.Tensor, neg_scores: torch.Tensor) -> float:
    """
    Compute AUROC where:
        pos_scores = scores for positive (ID) samples
        neg_scores = scores for negative (OOD) samples
    Larger scores mean more likely to be positive.
    """
    scores = torch.cat([pos_scores, neg_scores], dim=0)
    labels = torch.cat(
        [torch.ones_like(pos_scores, dtype=torch.long), torch.zeros_like(neg_scores, dtype=torch.long)],
        dim=0,
    )
    # Sort by decreasing score
    order = torch.argsort(scores, descending=True)
    labels = labels[order]

    # True/false positives as we sweep the threshold
    pos = (labels == 1).float()
    neg = (labels == 0).float()
    tps = torch.cumsum(pos, dim=0)
    fps = torch.cumsum(neg, dim=0)

    num_pos = pos.sum()
    num_neg = neg.sum()
    if num_pos == 0 or num_neg == 0:
        return float("nan")

    tpr = tps / num_pos
    fpr = fps / num_neg

    # Trapz over FPR (x-axis)
    auroc = torch.trapz(tpr, fpr).item()
    return float(auroc)


def _binary_aupr(pos_scores: torch.Tensor, neg_scores: torch.Tensor) -> float:
    """
    Compute AUPR where positive class is pos_scores.
    """
    scores = torch.cat([pos_scores, neg_scores], dim=0)
    labels = torch.cat(
        [torch.ones_like(pos_scores, dtype=torch.long), torch.zeros_like(neg_scores, dtype=torch.long)],
        dim=0,
    )
    order = torch.argsort(scores, descending=True)
    labels = labels[order]

    pos = (labels == 1).float()
    num_pos = pos.sum()
    if num_pos == 0:
        return float("nan")

    tps = torch.cumsum(pos, dim=0)
    fps = torch.cumsum(1.0 - pos, dim=0)
    precision = tps / (tps + fps + 1e-12)
    recall = tps / num_pos

    # AUPR is area under precision-recall curve (recall on x-axis)
    aupr = torch.trapz(precision, recall).item()
    return float(aupr)


def _fpr_at_tpr(pos_scores: torch.Tensor, neg_scores: torch.Tensor, target_tpr: float = 0.95) -> float:
    """
    Compute FPR at the smallest threshold where TPR >= target_tpr.
    """
    scores = torch.cat([pos_scores, neg_scores], dim=0)
    labels = torch.cat(
        [torch.ones_like(pos_scores, dtype=torch.long), torch.zeros_like(neg_scores, dtype=torch.long)],
        dim=0,
    )
    order = torch.argsort(scores, descending=True)
    labels = labels[order]

    pos = (labels == 1).float()
    neg = (labels == 0).float()
    tps = torch.cumsum(pos, dim=0)
    fps = torch.cumsum(neg, dim=0)

    num_pos = pos.sum()
    num_neg = neg.sum()
    if num_pos == 0 or num_neg == 0:
        return float("nan")

    tpr = tps / num_pos
    fpr = fps / num_neg

    idx = torch.nonzero(tpr >= target_tpr, as_tuple=False)
    if idx.numel() == 0:
        return 1.0
    return float(fpr[idx[0].item()])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="OOD detection on ImageNet with CLIP + sparse text encodings."
    )
    parser.add_argument(
        "--imagenet_root",
        type=str,
        required=True,
        help="Root of ImageNet (expects <root>/<split>/ with 1000 class folders).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["train", "val"],
        help="Which ImageNet split to use as ID data (default: val).",
    )
    parser.add_argument(
        "--imagenet_ood_root",
        type=str,
        required=True,
        help="Root of ImageNet OOD data (e.g., path to 'imagenetood' after extraction).",
    )
    parser.add_argument(
        "--class_index_path",
        type=str,
        default="resources/imagenet_class_index.json",
        help="Path to imagenet_class_index.json (already downloaded for zero-shot script).",
    )
    parser.add_argument(
        "--prompt_template",
        type=str,
        default="a photo of a {}.",
        help="Prompt template; {} will be replaced by the full ImageNet class label.",
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
        help="Number of atoms for OMP residual.",
    )
    parser.add_argument(
        "--max_dict_cos_sim",
        type=float,
        default=0.9,
        help="Maximum cosine similarity between a class embedding and dictionary atoms. "
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

    # 1) Load CLIP model and preprocessing
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name=args.model_name,
        pretrained=args.pretrained,
        device=device,
    )
    tokenizer = open_clip.get_tokenizer(args.model_name)
    model.eval()

    # 2) Load ID (ImageNet) and OOD datasets
    id_split_dir = os.path.join(args.imagenet_root, args.split)
    if not os.path.isdir(id_split_dir):
        raise RuntimeError(f"Expected ImageNet {args.split} directory at {id_split_dir}")
    id_dataset = datasets.ImageFolder(root=id_split_dir, transform=preprocess)
    id_loader = DataLoader(
        id_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    print(f"[ID] Loaded ImageNet '{args.split}' from {id_split_dir} with {len(id_dataset)} images.")

    ood_dataset = FlatImageDataset(root=args.imagenet_ood_root, transform=preprocess)
    ood_loader = DataLoader(
        ood_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    print(f"[OOD] Loaded OOD data from {args.imagenet_ood_root} with {len(ood_dataset)} images.")

    # 3) Build prompts and class embeddings (same as zero-shot script)
    class_index = izs.load_imagenet_class_index(args.class_index_path)
    wnid_to_label = izs.build_wnid_to_label_map(class_index)

    wnids: List[str] = list(id_dataset.classes)  # e.g., ["n01440764", ...]
    prompts = izs.build_prompts_for_imagenet_classes(
        wnids=wnids,
        wnid_to_label=wnid_to_label,
        template=args.prompt_template,
    )

    print(f"[prompts] Built {len(prompts)} prompts. Example:")
    for i in range(min(5, len(prompts))):
        print(f"  class {i:3d} ({wnids[i]}): {prompts[i]}")

    text_embs = izs.compute_text_embeddings(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        device=device,
    )

    sparse_text_embs = izs.precompute_sparse_text_embeddings(
        text_embs=text_embs,
        atoms=args.atoms,
        max_cos_sim=args.max_dict_cos_sim if 0.0 < args.max_dict_cos_sim < 1.0 else None,
    )

    # 4) Collect ID and OOD scores
    print("[scores] Collecting ID scores ...")
    id_scores_std, id_scores_sparse = collect_maxlogit_scores(
        model=model,
        dataloader=id_loader,
        text_embs=text_embs,
        sparse_text_embs=sparse_text_embs,
        device=device,
    )

    print("[scores] Collecting OOD scores ...")
    ood_scores_std, ood_scores_sparse = collect_maxlogit_scores(
        model=model,
        dataloader=ood_loader,
        text_embs=text_embs,
        sparse_text_embs=sparse_text_embs,
        device=device,
    )

    # 5) Compute OOD metrics
    print("\n=== OOD Detection Metrics (ID = ImageNet, OOD = imagenetood) ===")
    for name, id_s, ood_s in [
        ("Standard CLIP", id_scores_std, id_scores_sparse * 0 + ood_scores_std),  # placeholder
    ]:
        # We'll handle standard and sparse separately below; this placeholder loop is unused.
        pass

    # Standard CLIP
    auroc_std = _binary_auroc(id_scores_std, ood_scores_std)
    fpr95_std = _fpr_at_tpr(id_scores_std, ood_scores_std, target_tpr=0.95)
    aupr_in_std = _binary_aupr(id_scores_std, ood_scores_std)
    aupr_out_std = _binary_aupr(ood_scores_std, id_scores_std)

    # Sparse CLIP
    auroc_sparse = _binary_auroc(id_scores_sparse, ood_scores_sparse)
    fpr95_sparse = _fpr_at_tpr(id_scores_sparse, ood_scores_sparse, target_tpr=0.95)
    aupr_in_sparse = _binary_aupr(id_scores_sparse, ood_scores_sparse)
    aupr_out_sparse = _binary_aupr(ood_scores_sparse, id_scores_sparse)

    print("[Standard CLIP]")
    print(f"  AUROC          : {auroc_std:.4f}")
    print(f"  FPR@95%TPR     : {fpr95_std:.4f}")
    print(f"  AUPR (ID pos)  : {aupr_in_std:.4f}")
    print(f"  AUPR (OOD pos) : {aupr_out_std:.4f}")

    print("\n[Sparse (OMP residual)]")
    print(f"  AUROC          : {auroc_sparse:.4f}")
    print(f"  FPR@95%TPR     : {fpr95_sparse:.4f}")
    print(f"  AUPR (ID pos)  : {aupr_in_sparse:.4f}")
    print(f"  AUPR (OOD pos) : {aupr_out_sparse:.4f}")


if __name__ == "__main__":
    main()


