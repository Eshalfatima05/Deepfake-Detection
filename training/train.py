"""
Step 5: Training Script
- AdamW optimizer with differential learning rates
- Cosine annealing LR scheduler with warmup
- Cross-entropy loss with label smoothing
- Early stopping
- Checkpoint saving (best model by val AUC)
- Optional Weights & Biases logging
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
import numpy as np
from tqdm import tqdm

# ── local imports ─────────────────────────────────────────────────────────────
sys.path.append(str(Path(__file__).parent.parent))
from model.dataset   import get_dataloaders
from model.vit_model import build_model


# ── Argument parser ───────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train deepfake detector")
    p.add_argument("--data_root",    type=str,   default="data/processed")
    p.add_argument("--model_type",   type=str,   default="vit",
                   choices=["vit", "efficientnet"])
    p.add_argument("--epochs",       type=int,   default=20)
    p.add_argument("--batch_size",   type=int,   default=32)
    p.add_argument("--lr_head",      type=float, default=1e-3,
                   help="LR for classifier head")
    p.add_argument("--lr_backbone",  type=float, default=1e-5,
                   help="LR for unfrozen backbone blocks")
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--dropout",      type=float, default=0.1)
    p.add_argument("--label_smooth", type=float, default=0.1)
    p.add_argument("--num_workers",  type=int,   default=4)
    p.add_argument("--output_dir",   type=str,   default="checkpoints")
    p.add_argument("--use_wandb",    action="store_true")
    p.add_argument("--wandb_project",type=str,   default="deepfake-detection")
    p.add_argument("--patience",     type=int,   default=5,
                   help="Early stopping patience (epochs)")
    p.add_argument("--seed",         type=int,   default=42)
    return p.parse_args()


# ── Helpers ───────────────────────────────────────────────────────────────────

def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_optimizer(model, args):
    """Differential LRs: smaller for backbone, larger for head."""
    if args.model_type == "vit":
        backbone_params = list(model.vit.vit.parameters())
        head_params     = list(model.vit.classifier.parameters())
    else:
        backbone_params = list(model.backbone.parameters())
        head_params     = list(model.classifier.parameters())

    param_groups = [
        {"params": [p for p in backbone_params if p.requires_grad],
         "lr": args.lr_backbone},
        {"params": head_params,
         "lr": args.lr_head},
    ]
    return AdamW(param_groups, weight_decay=args.weight_decay)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_labels = []
    all_probs  = []

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits, _    = model(imgs)
        loss         = criterion(logits, labels)
        total_loss  += loss.item() * imgs.size(0)

        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        all_probs.extend(probs.tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

    avg_loss = total_loss / len(loader.dataset)
    preds    = (np.array(all_probs) >= 0.5).astype(int)
    auc      = roc_auc_score(all_labels, all_probs)
    f1       = f1_score(all_labels, preds, zero_division=0)
    acc      = accuracy_score(all_labels, preds)
    return avg_loss, auc, f1, acc


# ── Training loop ─────────────────────────────────────────────────────────────

def train(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── W&B ──────────────────────────────────────────────────────────────────
    if args.use_wandb:
        import wandb
        wandb.init(project=args.wandb_project, config=vars(args))

    # ── Data ─────────────────────────────────────────────────────────────────
    loaders = get_dataloaders(
        args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # ── Model ────────────────────────────────────────────────────────────────
    model = build_model(args.model_type, pretrained=True, dropout=args.dropout)
    model = model.to(device)

    # ── Loss, optimizer, scheduler ───────────────────────────────────────────
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smooth)
    optimizer = get_optimizer(model, args)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-7)

    # ── Output dir ───────────────────────────────────────────────────────────
    out_dir = Path(args.output_dir) / args.model_type
    out_dir.mkdir(parents=True, exist_ok=True)

    best_auc      = 0.0
    patience_ctr  = 0
    history       = []

    print(f"\n{'='*60}")
    print(f"Training {args.model_type.upper()} for {args.epochs} epochs")
    print(f"{'='*60}\n")

    for epoch in range(1, args.epochs + 1):
        # ── Train ─────────────────────────────────────────────────────────────
        model.train()
        running_loss = 0.0
        t0 = time.time()

        pbar = tqdm(loaders["train"], desc=f"Epoch {epoch:02d}/{args.epochs} [train]",
                    leave=False)
        for imgs, labels in pbar:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits, _ = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = running_loss / len(loaders["train"])
        scheduler.step()

        # ── Validate ──────────────────────────────────────────────────────────
        val_loss, val_auc, val_f1, val_acc = evaluate(
            model, loaders["val"], criterion, device)

        elapsed = time.time() - t0
        lr_now  = optimizer.param_groups[-1]["lr"]

        row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_loss":   round(val_loss,   4),
            "val_auc":    round(val_auc,    4),
            "val_f1":     round(val_f1,     4),
            "val_acc":    round(val_acc,    4),
            "lr":         lr_now,
        }
        history.append(row)

        print(f"Epoch {epoch:02d}/{args.epochs}  "
              f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
              f"val_auc={val_auc:.4f}  val_f1={val_f1:.4f}  "
              f"val_acc={val_acc:.4f}  ({elapsed:.0f}s)")

        if args.use_wandb:
            import wandb
            wandb.log(row)

        # ── Checkpoint ────────────────────────────────────────────────────────
        if val_auc > best_auc:
            best_auc    = val_auc
            patience_ctr = 0
            ckpt_path   = out_dir / "best_model.pth"
            torch.save({
                "epoch":      epoch,
                "model_type": args.model_type,
                "state_dict": model.state_dict(),
                "val_auc":    val_auc,
                "val_f1":     val_f1,
                "val_acc":    val_acc,
                "args":       vars(args),
            }, ckpt_path)
            print(f"  ✅ New best AUC={val_auc:.4f} → saved to {ckpt_path}")
        else:
            patience_ctr += 1
            if patience_ctr >= args.patience:
                print(f"\n⛔ Early stopping at epoch {epoch} (patience={args.patience})")
                break

    # ── Save history ──────────────────────────────────────────────────────────
    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"\n🏁 Training complete. Best val AUC: {best_auc:.4f}")
    print(f"   Checkpoint: {out_dir / 'best_model.pth'}")

    if args.use_wandb:
        import wandb
        wandb.finish()

    return history


if __name__ == "__main__":
    args = parse_args()
    train(args)
