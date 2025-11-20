import sys
import os
import argparse
import json
import re
import time
import numpy as np
import h5py
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import average_precision_score, precision_recall_curve
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

try:
    import nltk
    from nltk.corpus import wordnet as wn
except ImportError:
    pass

from legrad import LeWrapper
import open_clip
import sparse_encoding  # Import from scripts/sparse_encoding.py

def get_synset_name(wnid):
    try:
        # wnid is like 'n01322343'
        offset = int(wnid[1:])
        synset = wn.synset_from_pos_and_offset('n', offset)
        # Use the first lemma name, replace _ with space
        name = synset.lemmas()[0].name().replace('_', ' ')
        return name
    except Exception as e:
        # Fallback if wn lookup fails
        return wnid

def compute_iou_acc(heatmap, gt_mask, threshold=0.5):
    # heatmap: [H, W] in [0, 1]
    # gt_mask: [H, W] in {0, 1}
    
    pred_mask = (heatmap > threshold).astype(np.uint8)
    gt_mask = gt_mask.astype(np.uint8)
    
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    
    iou = intersection / (union + 1e-6)
    
    correct = (pred_mask == gt_mask).sum()
    total = gt_mask.size
    acc = correct / total
    
    return iou, acc

def compute_map_score(heatmap, gt_mask):
    # Average Precision
    # Flatten arrays
    y_true = gt_mask.flatten().astype(int)
    y_scores = heatmap.flatten()
    
    if y_true.sum() == 0:
        return 0.0 # No ground truth positive
        
    return average_precision_score(y_true, y_scores)

def main():
    parser = argparse.ArgumentParser(description='Benchmark LeGrad Segmentation')
    parser.add_argument('--mat_file', type=str, default='scripts/data/gtsegs_ijcv.mat')
    parser.add_argument('--limit', type=int, default=0, help='Limit number of images (0 for all)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--model_name', type=str, default='ViT-B-16')
    parser.add_argument('--pretrained', type=str, default='laion2b_s34b_b88k')
    parser.add_argument('--image_size', type=int, default=448)
    
    # Sparse settings
    parser.add_argument('--residual_atoms', type=int, default=8)
    parser.add_argument('--wn_use_siblings', type=int, default=1)
    
    args = parser.parse_args()
    
    # Load Model
    print(f"Loading model {args.model_name}...")
    model, _, _ = open_clip.create_model_and_transforms(model_name=args.model_name,
                                                        pretrained=args.pretrained,
                                                        device=args.device)
    tokenizer = open_clip.get_tokenizer(model_name=args.model_name)
    model.eval()
    model = LeWrapper(model, layer_index=0)
    
    # Download NLTK
    try:
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
    except:
        pass

    # Open Dataset
    print(f"Opening dataset {args.mat_file}...")
    try:
        f = h5py.File(args.mat_file, 'r')
        imgs_refs = f['value/img']
        gts_refs = f['value/gt']
        targets_refs = f['value/target']
        ids_refs = f['value/id']
        num_images = imgs_refs.shape[0]
    except Exception as e:
        print(f"Error opening dataset: {e}")
        return

    limit = args.limit if args.limit > 0 else num_images
    limit = min(limit, num_images)
    print(f"Processing {limit} images...")

    results = {
        'original': {'iou': [], 'acc': [], 'ap': []},
        'sparse': {'iou': [], 'acc': [], 'ap': []}
    }

    for idx in tqdm(range(limit)):
        try:
            # 1. Load Image
            img_ref = imgs_refs[idx, 0]
            img_obj = np.array(f[img_ref]) # (3, W, H) or (C, W, H) usually
            # Transpose to (H, W, C) for PIL
            # Based on inspection: numpy shape (3, 500, 407) -> transpose(2, 1, 0) -> (407, 500, 3)
            # This means H=407, W=500.
            img_np = img_obj.transpose(2, 1, 0)
            base_img = Image.fromarray(img_np)
            
            # Preprocess
            img_t = sparse_encoding.safe_preprocess(base_img, image_size=args.image_size).unsqueeze(0).to(args.device)
            
            # 2. Load GT
            gt_ref = gts_refs[idx, 0]
            gt_wrapper = f[gt_ref]
            if gt_wrapper.dtype == 'object':
                real_gt_ref = gt_wrapper[0,0]
                real_gt = np.array(f[real_gt_ref]) # (W, H) usually from Matlab
                gt_mask = real_gt.transpose(1, 0) # (H, W)
            else:
                # Should not happen based on inspection
                gt_mask = np.zeros((base_img.height, base_img.width))
            
            # Resize GT to image size if needed? 
            # The heatmaps will be [image_size, image_size]. 
            # We should resize heatmap to GT size or GT to heatmap size?
            # Usually resize heatmap to GT size (original image resolution) for evaluation.
            H_orig, W_orig = gt_mask.shape
            
            # 3. Get Class Name
            target_ref = targets_refs[idx, 0]
            target_data = np.array(f[target_ref])
            wnid = ''.join([chr(c) for c in target_data.flatten()])
            class_name = get_synset_name(wnid)
            
            prompt = f"a photo of a {class_name}."
            
            # 4. Compute Embeddings
            tok = tokenizer([prompt]).to(args.device)
            with torch.no_grad():
                original_1x = model.encode_text(tok, normalize=True)
            
            # --- ORIGINAL ---
            heatmap_orig = sparse_encoding.compute_map_for_embedding(model, img_t, original_1x) # [448, 448]
            # Resize to original size
            heatmap_orig_resized = F.interpolate(heatmap_orig.view(1, 1, args.image_size, args.image_size), 
                                                 size=(H_orig, W_orig), 
                                                 mode='bilinear', 
                                                 align_corners=False).squeeze().numpy()
            
            # --- SPARSE ---
            # Build neighbors
            tokens = re.findall(r'[a-z]+', prompt.lower())
            key = tokens[-1] if len(tokens) > 0 else ''
            
            wl = []
            if key:
                wl = sparse_encoding.wordnet_neighbors_configured(
                    key,
                    use_synonyms=False,
                    use_hypernyms=False,
                    use_hyponyms=False,
                    use_siblings=bool(args.wn_use_siblings),
                    limit_per_relation=8
                )
            
            # Build dictionary D
            D = None
            if len(wl) > 0:
                ext_emb = sparse_encoding.build_wordlist_neighbors_embedding(tokenizer, model, wl, args.device)
                D = F.normalize(ext_emb, dim=-1)
            
            sparse_1x = sparse_encoding.omp_sparse_residual(original_1x, D, max_atoms=args.residual_atoms)
            
            heatmap_sparse = sparse_encoding.compute_map_for_embedding(model, img_t, sparse_1x)
            heatmap_sparse_resized = F.interpolate(heatmap_sparse.view(1, 1, args.image_size, args.image_size), 
                                                   size=(H_orig, W_orig), 
                                                   mode='bilinear', 
                                                   align_corners=False).squeeze().numpy()
            
            # --- METRICS ---
            # Original
            iou_o, acc_o = compute_iou_acc(heatmap_orig_resized, gt_mask)
            ap_o = compute_map_score(heatmap_orig_resized, gt_mask)
            
            results['original']['iou'].append(iou_o)
            results['original']['acc'].append(acc_o)
            results['original']['ap'].append(ap_o)
            
            # Sparse
            iou_s, acc_s = compute_iou_acc(heatmap_sparse_resized, gt_mask)
            ap_s = compute_map_score(heatmap_sparse_resized, gt_mask)
            
            results['sparse']['iou'].append(iou_s)
            results['sparse']['acc'].append(acc_s)
            results['sparse']['ap'].append(ap_s)
            
        except Exception as e:
            print(f"Error processing idx {idx}: {e}")
            continue

    # Summary
    print("\n--- Results ---")
    for method in ['original', 'sparse']:
        miou = np.mean(results[method]['iou']) * 100
        macc = np.mean(results[method]['acc']) * 100
        map_score = np.mean(results[method]['ap']) * 100
        print(f"{method.capitalize()}: mIoU={miou:.2f}, PixelAcc={macc:.2f}, mAP={map_score:.2f}")

if __name__ == '__main__':
    main()

