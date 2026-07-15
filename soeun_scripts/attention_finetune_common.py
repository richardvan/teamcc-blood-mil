#!/usr/bin/env python3
"""
attention_finetune_common.py — Attention-MIL + ResNet50 backbone fine-tuning

팀원의 06a_train_cnn_cv.py 방식(이미지 직접 로드, layer4 unfreeze, SGD 이중 학습률,
instances_per_step 샘플링)을 따르되, pooling을 두 종류로 지원합니다:

  - AttentionMILFineTune         : gated attention (Ilse et al. 2018, 클래스 공유)
  - ClassWiseAttentionMILFineTune: class-wise attention (SCEMILA 방식, 클래스별 독립)

두 모델 다 backbone(ResNet50)+encoder는 공유 구조(MILImageModel)이고,
pooling/classifier 부분만 다릅니다. train_one_step / predict_patient 등
학습·추론 함수는 두 모델에 공용으로 씁니다 (모델이 forward(images)->(logits, attn)
인터페이스만 지키면 되므로).

가장 중요한 차이 (frozen feature 버전과):
  - cache/cnn_features/*.pt 를 안 씁니다. 이미지 원본을 매번 backbone에 통과시켜야
    pooling까지 end-to-end로 학습/역전파가 가능하기 때문입니다.
  - 그래서 학습이 훨씬 느립니다 (매 epoch마다 이미지 → CNN forward+backward).
"""

import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

from mil_common import log

IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

RESNET_BLOCKS = ["layer1", "layer2", "layer3", "layer4"]


# ──────────────────────────────────────────────────────────────
# 이미지 로딩 유틸
# ──────────────────────────────────────────────────────────────

def list_patient_image_paths(organized_dir: Path, patient_id: str, image_ext: str = ".tif"):
    return sorted((organized_dir / patient_id).glob(f"*{image_ext}"))


def load_images_as_tensor(paths, device):
    imgs = [TRANSFORM(Image.open(p).convert("RGB")) for p in paths]
    return torch.stack(imgs).to(device)


# ──────────────────────────────────────────────────────────────
# 공용 베이스: fine-tunable ResNet50 backbone + encoder
# ──────────────────────────────────────────────────────────────

class MILImageModel(nn.Module):
    """
    backbone(ResNet50, 일부 unfreeze) + encoder(2048→hidden) 는 공유.
    pooling/classifier(pool_and_classify)만 서브클래스가 구현.
    """
    def __init__(self, hidden_dim: int = 256, dropout: float = 0.3, unfreeze_from: str = "layer4"):
        super().__init__()
        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.unfreeze_from = unfreeze_from
        self._apply_unfreeze(unfreeze_from)

        self.encoder = nn.Sequential(
            nn.Linear(2048, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
        )

    def _apply_unfreeze(self, unfreeze_from: str):
        for p in self.backbone.parameters():
            p.requires_grad = False
        if unfreeze_from == "none":
            return
        if unfreeze_from == "all":
            for p in self.backbone.parameters():
                p.requires_grad = True
            return
        start = RESNET_BLOCKS.index(unfreeze_from)
        for block_name in RESNET_BLOCKS[start:]:
            for p in getattr(self.backbone, block_name).parameters():
                p.requires_grad = True

    def backbone_trainable_params(self):
        return [p for p in self.backbone.parameters() if p.requires_grad]

    def extract_h(self, images: torch.Tensor) -> torch.Tensor:
        """images: (N,3,224,224) -> h: (N, hidden)"""
        feats = self.backbone(images)
        return self.encoder(feats)

    def pool_and_classify(self, h: torch.Tensor):
        """서브클래스가 구현: h(N,hidden) -> logits(1,n_classes), attn"""
        raise NotImplementedError

    def head_params(self):
        """서브클래스가 구현: encoder 이후(attention+classifier) 파라미터 리스트"""
        raise NotImplementedError

    def forward(self, images: torch.Tensor):
        h = self.extract_h(images)
        return self.pool_and_classify(h)


# ──────────────────────────────────────────────────────────────
# 모델 A: Gated Attention (클래스 공유) — Ilse et al. 2018
# ──────────────────────────────────────────────────────────────

class AttentionMILFineTune(MILImageModel):
    def __init__(self, hidden_dim: int = 256, attn_dim: int = 128, n_classes: int = 5,
                 dropout: float = 0.3, gated: bool = True, unfreeze_from: str = "layer4"):
        super().__init__(hidden_dim=hidden_dim, dropout=dropout, unfreeze_from=unfreeze_from)
        self.gated = gated
        self.attn_V = nn.Sequential(nn.Linear(hidden_dim, attn_dim), nn.Tanh())
        if gated:
            self.attn_U = nn.Sequential(nn.Linear(hidden_dim, attn_dim), nn.Sigmoid())
        self.attn_w = nn.Linear(attn_dim, 1)
        self.classifier = nn.Linear(hidden_dim, n_classes)

    def head_params(self):
        params = list(self.encoder.parameters()) + list(self.attn_V.parameters()) \
                  + list(self.attn_w.parameters()) + list(self.classifier.parameters())
        if self.gated:
            params += list(self.attn_U.parameters())
        return params

    def pool_and_classify(self, h: torch.Tensor):
        a_v = self.attn_V(h)
        if self.gated:
            a_u = self.attn_U(h)
            scores = self.attn_w(a_v * a_u)
        else:
            scores = self.attn_w(a_v)
        attn = torch.softmax(scores, dim=0)          # (N,1)
        z = (attn * h).sum(dim=0, keepdim=True)        # (1,hidden)
        logits = self.classifier(z)                    # (1,n_classes)
        return logits, attn.squeeze(1)


# ──────────────────────────────────────────────────────────────
# 모델 B: Class-wise Attention (SCEMILA 방식, 클래스별 독립)
# ──────────────────────────────────────────────────────────────

class ClassWiseAttentionMILFineTune(MILImageModel):
    def __init__(self, hidden_dim: int = 256, attn_dim: int = 128, n_classes: int = 5,
                 dropout: float = 0.3, unfreeze_from: str = "layer4"):
        super().__init__(hidden_dim=hidden_dim, dropout=dropout, unfreeze_from=unfreeze_from)
        self.n_classes = n_classes
        self.attn_dim = attn_dim
        self.V = nn.Linear(hidden_dim, n_classes * attn_dim, bias=False)
        self.w = nn.Parameter(torch.randn(n_classes, attn_dim) * 0.01)
        self.classifier = nn.Linear(hidden_dim, 1)   # 클래스 간 공유

    def head_params(self):
        return list(self.encoder.parameters()) + [self.w] + list(self.V.parameters()) \
                + list(self.classifier.parameters())

    def pool_and_classify(self, h: torch.Tensor):
        N = h.shape[0]
        v = torch.tanh(self.V(h)).view(N, self.n_classes, self.attn_dim)   # (N,C,A)
        scores = torch.einsum("nca,ca->nc", v, self.w)                     # (N,C)
        attn = torch.softmax(scores, dim=0)                                 # (N,C), 클래스별 독립 정규화
        z = torch.einsum("nc,nh->ch", attn, h)                              # (C,hidden)
        logits = self.classifier(z).squeeze(-1).unsqueeze(0)               # (1,C)
        return logits, attn   # attn: (N, n_classes) — 클래스별 attention 전부 보존


# ──────────────────────────────────────────────────────────────
# 학습 1-step (모델 종류 무관, forward(images) 인터페이스만 있으면 됨)
# ──────────────────────────────────────────────────────────────

def train_one_step(model, organized_dir, patient_id, label, image_ext,
                    instances_per_step, rng: random.Random, device, optimizer, criterion):
    paths = list_patient_image_paths(organized_dir, patient_id, image_ext)
    if len(paths) > instances_per_step:
        paths = rng.sample(paths, instances_per_step)
    images = load_images_as_tensor(paths, device)
    target = torch.tensor([label], device=device)

    optimizer.zero_grad()
    logits, _ = model(images)
    loss = criterion(logits, target)
    loss.backward()
    optimizer.step()

    correct = int(logits.argmax(dim=1).item() == label)
    return loss.item(), correct


@torch.no_grad()
def predict_patient(model, organized_dir, patient_id, image_ext, device, batch_size: int = 64):
    """
    backbone forward는 batch_size씩 나눠서(메모리 절약), pooling은 전체 인스턴스에 대해 한 번에.
    반환: logits (1, n_classes), attn (gated: (N,) / class-wise: (N, n_classes))
    """
    model.eval()
    paths = list_patient_image_paths(organized_dir, patient_id, image_ext)
    h_chunks = []
    for i in range(0, len(paths), batch_size):
        images = load_images_as_tensor(paths[i:i + batch_size], device)
        h_chunks.append(model.extract_h(images))
    h = torch.cat(h_chunks, dim=0)
    logits, attn = model.pool_and_classify(h)
    return logits, attn


@torch.no_grad()
def compute_val_loss(model, organized_dir, patient_ids, labels, image_ext, device, criterion):
    if not patient_ids:
        return None
    total = 0.0
    for pid in patient_ids:
        logits, _ = predict_patient(model, organized_dir, pid, image_ext, device)
        target = torch.tensor([labels[pid]], device=device)
        total += criterion(logits, target).item()
    return total / len(patient_ids)


@torch.no_grad()
def evaluate_patients(model, organized_dir, patient_ids, labels, image_ext, device):
    y_true, y_pred, y_proba = [], [], []
    for pid in patient_ids:
        logits, _ = predict_patient(model, organized_dir, pid, image_ext, device)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        y_true.append(labels[pid])
        y_pred.append(int(np.argmax(probs)))
        y_proba.append(probs)
    return np.array(y_true), np.array(y_pred), np.array(y_proba)


# ──────────────────────────────────────────────────────────────
# 학습 루프 (fold별 또는 최종 모델 공용, 모델 종류 무관)
# ──────────────────────────────────────────────────────────────

def train_model_finetune(model, organized_dir, train_ids, val_ids, labels, image_ext,
                          device, max_epochs, lr_head, lr_backbone, weight_decay,
                          class_weights, instances_per_step=32, patience=15,
                          seed=42, verbose=True, use_early_stopping=True):
    """
    val_ids=[] 이면 "최종 모델" 모드: val 없이 max_epochs 만큼 고정 학습.
    반환: (model, best_val_f1 또는 None, best_epoch, train_loss_history)
    """
    from sklearn.metrics import f1_score

    model.to(device)
    rng = random.Random(seed)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.SGD([
        {"params": model.head_params(), "lr": lr_head},
        {"params": model.backbone_trainable_params(), "lr": lr_backbone},
    ], weight_decay=weight_decay)

    best_val_f1, best_val_loss = -1.0, float("inf")
    best_state, best_epoch, no_improve = None, max_epochs, 0
    train_loss_history = []

    for epoch in range(1, max_epochs + 1):
        model.train()
        shuffled = train_ids[:]
        rng.shuffle(shuffled)

        epoch_loss = 0.0
        for pid in shuffled:
            loss, _ = train_one_step(
                model, organized_dir, pid, labels[pid], image_ext,
                instances_per_step, rng, device, optimizer, criterion,
            )
            epoch_loss += loss
        avg_train_loss = epoch_loss / len(shuffled)
        train_loss_history.append(avg_train_loss)

        if val_ids:
            val_loss = compute_val_loss(model, organized_dir, val_ids, labels, image_ext, device, criterion)
            y_true, y_pred, _ = evaluate_patients(model, organized_dir, val_ids, labels, image_ext, device)
            val_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                best_epoch = epoch

            if val_loss < best_val_loss - 1e-4:
                best_val_loss = val_loss
                no_improve = 0
            else:
                no_improve += 1

            if verbose:
                log(f"    epoch {epoch:3d}/{max_epochs} | train_loss {avg_train_loss:.4f} "
                    f"| val_loss {val_loss:.4f} | val_f1_macro {val_f1:.3f} "
                    f"| best_f1 {best_val_f1:.3f} | patience {no_improve}/{patience}")

            if use_early_stopping and no_improve >= patience:
                if verbose:
                    log(f"    early stopping @ epoch {epoch}")
                break
        else:
            if verbose and (epoch % 5 == 0 or epoch == 1):
                log(f"    epoch {epoch:3d}/{max_epochs} | train_loss {avg_train_loss:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, (best_val_f1 if val_ids else None), best_epoch, train_loss_history
