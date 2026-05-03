"""
Step 6: Evaluation Script
- AUC-ROC, F1, Accuracy, Precision, Recall
- Confusion matrix
- ROC curve plot
- Attention Rollout visualization
- Cross-dataset evaluation (Celeb-DF)
"""

import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image
from sklearn.metrics import (
    roc_auc_score, f1_score, accuracy_score,
    precision_score, recall_score,
    confusion_matrix, roc_curve, classification_report
)

sys.path.append(str(Path(__file__).parent.parent))
from model.dataset   import DeepfakeDataset, get_val_transforms, denormalize
from model.vit_model import build_model


# ── Args ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",  type=str, required=True,
                   help="Path to best_model.pth")
    p.add_argument("--data_root",   type=str, default="data/processed",
                   help="Root with test/ subfolder")
    p.add_argument("--output_dir",  type=str, default="evaluation_results")
    p.add_argument("--batch_size",  type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--num_attn_samples", type=int, default=8,
                   help="Number of images to generate attention maps for")
    return p.parse_args()


# ── Attention Rollout ─────────────────────────────────────────────────────────

def attention_rollout(attentions, discard_ratio: float = 0.9):
    """
    Compute attention rollout from a list of attention weight tensors.
    attentions: tuple of (1, num_heads, seq_len, seq_len)
    Returns: (14, 14) attention map for the [CLS] token
    """
    result = torch.eye(attentions[0].size(-1))
    for attn in attentions:
        attn = attn.squeeze(0).mean(dim=0)   # (seq_len, seq_len)
        flat = attn.view(-1)
        threshold = flat.kthvalue(int(flat.size(0) * discard_ratio)).values
        attn = torch.where(attn > threshold, attn, torch.zeros_like(attn))
        attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-8)
        attn = attn + torch.eye(attn.size(-1))
        attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-8)
        result = torch.matmul(attn, result)

    # CLS token attention to all patches
    mask = result[0, 1:]   # skip CLS token itself → (196,)
    width = int(mask.size(0) ** 0.5)  # 14
    mask  = mask.reshape(width, width)
    mask  = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
    return mask.numpy()


def visualize_attention(model, dataset, device, out_dir: Path, n_samples: int = 8):
    """Save attention rollout overlay images."""
    model.eval()
    indices = np.random.choice(len(dataset), min(n_samples, len(dataset)), replace=False)
    fig, axes = plt.subplots(n_samples, 3, figsize=(12, 4 * n_samples))
    if n_samples == 1:
        axes = [axes]

    for row_idx, idx in enumerate(indices):
        img_tensor, label = dataset[idx]
        inp = img_tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            logits, attentions = model(inp)
            prob_fake = torch.softmax(logits, dim=1)[0, 1].item()

        # Original image (denormalized)
        orig_np = denormalize(img_tensor).permute(1, 2, 0).numpy()

        # Attention map
        attn_map = attention_rollout(attentions)
        attn_resized = np.array(
            Image.fromarray((attn_map * 255).astype(np.uint8)).resize(
                (224, 224), Image.BILINEAR
            )
        ) / 255.0

        # Overlay
        overlay = orig_np.copy()
        heatmap = plt.cm.jet(attn_resized)[..., :3]
        overlay = 0.55 * orig_np + 0.45 * heatmap

        true_label = "REAL" if label == 0 else "FAKE"
        pred_label = "FAKE" if prob_fake >= 0.5 else "REAL"
        color      = "green" if true_label == pred_label else "red"

        axes[row_idx][0].imshow(orig_np)
        axes[row_idx][0].set_title(f"Original\nTrue: {true_label}", fontsize=10)
        axes[row_idx][0].axis("off")

        axes[row_idx][1].imshow(attn_map, cmap="jet")
        axes[row_idx][1].set_title("Attention Rollout", fontsize=10)
        axes[row_idx][1].axis("off")

        axes[row_idx][2].imshow(np.clip(overlay, 0, 1))
        axes[row_idx][2].set_title(
            f"Overlay\nPred: {pred_label} ({prob_fake:.2f})",
            fontsize=10, color=color
        )
        axes[row_idx][2].axis("off")

    plt.tight_layout()
    save_path = out_dir / "attention_rollout.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Attention maps saved → {save_path}")


# ── Main evaluation ───────────────────────────────────────────────────────────

def evaluate(args):
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load checkpoint ───────────────────────────────────────────────────────
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt       = torch.load(args.checkpoint, map_location=device)
    model_type = ckpt.get("model_type", "vit")
    model      = build_model(model_type, pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model      = model.to(device)
    model.eval()
    print(f"Checkpoint loaded — val AUC at save time: {ckpt.get('val_auc', '?'):.4f}")

    # ── Test dataset ──────────────────────────────────────────────────────────
    test_ds = DeepfakeDataset(
        Path(args.data_root) / "test",
        transform=get_val_transforms()
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # ── Inference ─────────────────────────────────────────────────────────────
    all_labels = []
    all_probs  = []

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs   = imgs.to(device)
            logits, _ = model(imgs)
            probs  = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            all_probs.extend(probs.tolist())
            all_labels.extend(labels.numpy().tolist())

    all_labels = np.array(all_labels)
    all_probs  = np.array(all_probs)
    all_preds  = (all_probs >= 0.5).astype(int)

    # ── Metrics ───────────────────────────────────────────────────────────────
    auc  = roc_auc_score(all_labels, all_probs)
    f1   = f1_score(all_labels, all_preds, zero_division=0)
    acc  = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec  = recall_score(all_labels, all_preds, zero_division=0)

    metrics = {
        "AUC-ROC":   round(float(auc),  4),
        "F1-Score":  round(float(f1),   4),
        "Accuracy":  round(float(acc),  4),
        "Precision": round(float(prec), 4),
        "Recall":    round(float(rec),  4),
    }

    print("\n" + "="*45)
    print("         TEST SET EVALUATION RESULTS")
    print("="*45)
    for k, v in metrics.items():
        print(f"  {k:<12}: {v:.4f}")
    print("="*45)
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds,
                                target_names=["Real", "Fake"]))

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # ── Confusion matrix ──────────────────────────────────────────────────────
    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Real", "Fake"],
                yticklabels=["Real", "Fake"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix — {model_type.upper()}")
    plt.tight_layout()
    plt.savefig(out_dir / "confusion_matrix.png", dpi=150)
    plt.close()
    print(f"Confusion matrix → {out_dir / 'confusion_matrix.png'}")

    # ── ROC curve ─────────────────────────────────────────────────────────────
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#1a1a2e", lw=2, label=f"AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve — {model_type.upper()} Deepfake Detector")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_dir / "roc_curve.png", dpi=150)
    plt.close()
    print(f"ROC curve        → {out_dir / 'roc_curve.png'}")

    # ── Attention rollout ─────────────────────────────────────────────────────
    if model_type == "vit":
        print(f"\nGenerating attention maps for {args.num_attn_samples} samples...")
        visualize_attention(model, test_ds, device, out_dir, n_samples=args.num_attn_samples)

    print(f"\n✅ All evaluation outputs saved to: {out_dir}/")
    return metrics


if __name__ == "__main__":
    args = parse_args()
    evaluate(args)
