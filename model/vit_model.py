"""
Step 4: Model — Vision Transformer (ViT-B/16)
- Loads pretrained ViT-B/16 from HuggingFace
- Freezes first 8 encoder blocks
- Replaces classification head with 2-class linear layer
- Also defines EfficientNet-B4 baseline for comparison
"""

import torch
import torch.nn as nn
from transformers import ViTForImageClassification, ViTConfig


# ── ViT-B/16 Deepfake Detector ────────────────────────────────────────────────

class ViTDeepfakeDetector(nn.Module):
    """
    Fine-tuned ViT-B/16 for binary deepfake detection.

    Architecture:
      - Backbone : ViT-B/16 pretrained on ImageNet-21k
      - Frozen   : patch embeddings + first 8 transformer blocks
      - Trainable: last 4 transformer blocks + LayerNorm + classifier head
      - Head     : Linear(768 → 2)
    """

    NUM_BLOCKS_TO_FREEZE = 8

    def __init__(self, pretrained: bool = True, dropout: float = 0.1):
        super().__init__()

        model_name = "google/vit-base-patch16-224-in21k"

        if pretrained:
            self.vit = ViTForImageClassification.from_pretrained(
                model_name,
                num_labels=2,
                ignore_mismatched_sizes=True,
            )
        else:
            config = ViTConfig(
                hidden_size=768,
                num_hidden_layers=12,
                num_attention_heads=12,
                intermediate_size=3072,
                num_labels=2,
            )
            self.vit = ViTForImageClassification(config)

        # Replace classification head
        in_features = self.vit.config.hidden_size  # 768
        self.vit.classifier = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Dropout(dropout),
            nn.Linear(in_features, 256),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(256, 2),
        )

        self._freeze_layers()

    def _freeze_layers(self):
        """Freeze patch embeddings and first N transformer blocks."""
        # Freeze embeddings
        for param in self.vit.vit.embeddings.parameters():
            param.requires_grad = False

        # Freeze first NUM_BLOCKS_TO_FREEZE encoder layers
        encoder_layers = self.vit.vit.encoder.layer
        for i, layer in enumerate(encoder_layers):
            if i < self.NUM_BLOCKS_TO_FREEZE:
                for param in layer.parameters():
                    param.requires_grad = False

    def unfreeze_all(self):
        """Call this for fine-tuning stage 2 (optional)."""
        for param in self.parameters():
            param.requires_grad = True

    def forward(self, pixel_values: torch.Tensor):
        """
        Args:
            pixel_values: (B, 3, 224, 224) normalized tensor
        Returns:
            logits: (B, 2)
        """
        outputs = self.vit(pixel_values=pixel_values, output_attentions=True)
        return outputs.logits, outputs.attentions

    def get_attention_maps(self, pixel_values: torch.Tensor):
        """Return attention weights from all 12 heads for visualization."""
        with torch.no_grad():
            outputs = self.vit(pixel_values=pixel_values, output_attentions=True)
        return outputs.attentions   # tuple of (B, num_heads, seq_len, seq_len)

    def count_parameters(self):
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable


# ── EfficientNet-B4 Baseline ──────────────────────────────────────────────────

class EfficientNetBaseline(nn.Module):
    """
    EfficientNet-B4 baseline for comparison with ViT.
    """
    def __init__(self, pretrained: bool = True, dropout: float = 0.3):
        super().__init__()
        import timm
        self.backbone = timm.create_model(
            "efficientnet_b4",
            pretrained=pretrained,
            num_classes=0,       # remove original head
            global_pool="avg",
        )
        in_features = self.backbone.num_features  # 1792
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(256, 2),
        )

    def forward(self, x: torch.Tensor):
        features = self.backbone(x)
        logits   = self.classifier(features)
        return logits, None   # None = no attention maps


# ── Factory function ──────────────────────────────────────────────────────────

def build_model(model_type: str = "vit", pretrained: bool = True, dropout: float = 0.1):
    if model_type == "vit":
        model = ViTDeepfakeDetector(pretrained=pretrained, dropout=dropout)
        total, trainable = model.count_parameters()
        print(f"ViT-B/16 — Total params: {total:,} | Trainable: {trainable:,} "
              f"({100 * trainable / total:.1f}%)")
        return model
    elif model_type == "efficientnet":
        model = EfficientNetBaseline(pretrained=pretrained, dropout=dropout)
        total = sum(p.numel() for p in model.parameters())
        train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"EfficientNet-B4 — Total: {total:,} | Trainable: {train:,}")
        return model
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Choose 'vit' or 'efficientnet'.")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = build_model("vit").to(device)
    dummy  = torch.randn(2, 3, 224, 224).to(device)
    logits, attns = model(dummy)
    print(f"Output logits shape : {logits.shape}")
    print(f"Attention maps      : {len(attns)} layers, each {attns[0].shape}")
