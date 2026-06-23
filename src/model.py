import torch
import torch.nn as nn
from torchvision import models


# =====================================================
# BASELINE MODEL — chỉ nhận ảnh
# =====================================================

class BaselineModel(nn.Module):
    """
    ResNet50 pretrained.
    Freeze: layer1, layer2, layer3
    Unfreeze: layer4 + fc
    Output: logits (batch, 8) — KHÔNG qua Softmax
    """
    def __init__(self, num_classes=8):
        super().__init__()

        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

        # Freeze layer1 → layer3
        for name, param in backbone.named_parameters():
            if any(name.startswith(layer) for layer in ["layer1", "layer2", "layer3"]):
                param.requires_grad = False

        # Bỏ fc gốc, giữ phần còn lại
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])  # output: (batch, 2048, 1, 1)

        # Classifier mới
        self.classifier = nn.Linear(2048, num_classes)

    def forward(self, img):
        x = self.backbone(img)          # (batch, 2048, 1, 1)
        x = x.flatten(1)               # (batch, 2048)
        return self.classifier(x)      # (batch, 8)


# =====================================================
# MULTIMODAL MODEL — ảnh + metadata
# =====================================================

class MultimodalNet(nn.Module):
    """
    CNN Branch : ResNet50 → 2048 dim
    MLP Branch : meta_dim → 64 → 16 dim
    Fusion     : concat → 2064 → Linear(512) → BN → ReLU → Dropout(0.3) → Linear(8)
    Output     : logits (batch, 8) — KHÔNG qua Softmax
    """
    def __init__(self, meta_dim: int, num_classes=8):
        super().__init__()

        # --- CNN Branch ---
        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

        for name, param in backbone.named_parameters():
            if any(name.startswith(layer) for layer in ["layer1", "layer2", "layer3"]):
                param.requires_grad = False

        self.cnn = nn.Sequential(*list(backbone.children())[:-1])  # (batch, 2048, 1, 1)

        # --- MLP Branch ---
        self.mlp = nn.Sequential(
            nn.Linear(meta_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 16)
        )

# --- Fusion Classifier ---
        self.classifier = nn.Sequential(
            nn.Linear(2048 + 16, 512),  # 2064 → 512
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)  # 512 → 8
        )

    def forward(self, img, meta):
        # CNN branch
        img_feat = self.cnn(img).flatten(1)   # (batch, 2048)

        # MLP branch
        meta_feat = self.mlp(meta)            # (batch, 16)

        # Concat + classify
        fused = torch.cat([img_feat, meta_feat], dim=1)  # (batch, 2064)
        return self.classifier(fused)                     # (batch, 8)


# =====================================================
# HELPER — load model để inference
# =====================================================

def load_model(model_path: str, meta_dim: int, num_classes=8, device="cpu") -> MultimodalNet:
    """
    Load MultimodalNet từ checkpoint .pth để inference.
    Dùng trong app.py — KHÔNG dùng trong train.py.
    """
    model = MultimodalNet(meta_dim=meta_dim, num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model.to(device)


# =====================================================
# SANITY CHECK
# =====================================================

if __name__ == "__main__":
    BATCH = 4
    META_DIM = 20  # thay bằng len(feature_cols) thực tế sau preprocessing

    dummy_img  = torch.randn(BATCH, 3, 224, 224)
    dummy_meta = torch.randn(BATCH, META_DIM)

    # --- Test BaselineModel ---
    baseline = BaselineModel(num_classes=8)
    out = baseline(dummy_img)
    assert out.shape == (BATCH, 8), f"Baseline output sai: {out.shape}"
    print(f"[OK] BaselineModel output : {out.shape}")

    # --- Test MultimodalNet ---
    multimodal = MultimodalNet(meta_dim=META_DIM, num_classes=8)
    out = multimodal(dummy_img, dummy_meta)
    assert out.shape == (BATCH, 8), f"Multimodal output sai: {out.shape}"
    print(f"[OK] MultimodalNet output : {out.shape}")

    # --- Kiểm tra freeze ---
    frozen = [n for n, p in multimodal.named_parameters() if not p.requires_grad]
    print(f"[OK] Frozen params        : {len(frozen)} tensors")

    print("\n===== model.py PASSED =====")