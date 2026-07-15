#!/usr/bin/env python3
"""
cnn_mil_common.py — CNN-MIL(mean/max pooling) 모델 정의 + 학습/평가 유틸

attention_mil_common.py 와 구조를 맞춘 CNN-MIL 버전입니다.
ResNet50 은 frozen 특징 추출기로만 쓰고(00_extract_cnn_features.py 캐시 사용),
그 위에 mean/max pooling + 얕은 MLP로 분류합니다 — attention pooling이 없다는 것만
Attention-MIL과 다르고 나머지 학습/평가/저장 방식은 동일합니다.
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, accuracy_score

from mil_common import log, BagObject, compute_class_weights


# ──────────────────────────────────────────────────────────────
# 모델 정의
# ──────────────────────────────────────────────────────────────

class CNNMIL(nn.Module):
    """
    pooling='mean' | 'max'  : 인스턴스 임베딩을 encoder에 태운 뒤 pooling
    pooling='instance_max'  : 인스턴스별로 먼저 분류하고 로짓을 max-pool
    """
    def __init__(self, in_dim: int, hidden_dim: int, n_classes: int,
                 pooling: str = "mean", dropout: float = 0.3):
        super().__init__()
        assert pooling in ("mean", "max", "instance_max")
        self.pooling = pooling

        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(hidden_dim, n_classes)

    def encode_and_pool(self, bag: torch.Tensor):
        """h(인스턴스별 인코딩), z(pooled bag 벡터) 반환. instance_max는 z=None."""
        h = self.encoder(bag)  # (N, hidden)
        if self.pooling == "instance_max":
            return h, None
        z = h.mean(dim=0, keepdim=True) if self.pooling == "mean" else h.max(dim=0, keepdim=True)[0]
        return h, z

    def forward(self, bag: torch.Tensor):
        h, z = self.encode_and_pool(bag)
        if self.pooling == "instance_max":
            inst_logits = self.classifier(h)
            logits = inst_logits.max(dim=0, keepdim=True)[0]
            return logits
        return self.classifier(z)


class CNNMILWrapper:
    """shared_functions_V2 이 요구하는 predict_bag() 인터페이스."""
    def __init__(self, model: nn.Module, device: str):
        self.model = model.eval()
        self.device = device

    @torch.no_grad()
    def predict_bag(self, bag_instances) -> dict:
        x = bag_instances.to(self.device)
        if x.dtype != torch.float32:
            x = x.float()
        logits = self.model(x)
        probs = torch.softmax(logits, dim=1)[0]
        pred_label = int(torch.argmax(probs).item())
        pred_score = float(probs[pred_label].item())
        return {"pred_score": pred_score, "pred_label": pred_label}

    @torch.no_grad()
    def predict_bag_proba(self, bag_instances) -> np.ndarray:
        x = bag_instances.to(self.device)
        if x.dtype != torch.float32:
            x = x.float()
        logits = self.model(x)
        return torch.softmax(logits, dim=1)[0].cpu().numpy()

    @torch.no_grad()
    def get_bag_vector(self, bag_instances) -> np.ndarray:
        """pooled bag 벡터 z (분류기 직전) — PCA 시각화용. instance_max는 지원 안 함."""
        x = bag_instances.to(self.device)
        if x.dtype != torch.float32:
            x = x.float()
        _, z = self.model.encode_and_pool(x)
        if z is None:
            raise ValueError("instance_max pooling은 bag 벡터가 없어 PCA에 쓸 수 없습니다.")
        return z.cpu().numpy().flatten()


# ──────────────────────────────────────────────────────────────
# 학습/평가 루프 (attention_mil_common.py와 동일한 인터페이스)
# ──────────────────────────────────────────────────────────────

def evaluate(model, bags, device):
    model.eval()
    y_true, y_pred, y_proba = [], [], []
    with torch.no_grad():
        for bag in bags:
            x = bag.instances.to(device).float()
            logits = model(x)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
            pred = int(np.argmax(probs))
            y_true.append(bag.true_label)
            y_pred.append(pred)
            y_proba.append(probs)
    y_true, y_pred, y_proba = np.array(y_true), np.array(y_pred), np.array(y_proba)
    acc = accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, average="macro", zero_division=0)
    return acc, f1m, y_true, y_pred, y_proba


def train_model(model, train_bags, val_bags, device, epochs, lr, weight_decay,
                 class_weights, patience=15, verbose=True, use_early_stopping=True):
    """
    val_bags=None 이면 "최종 모델" 모드: val 없이 고정 epochs만 학습하고
    마지막 시점의 가중치를 그대로 씁니다 (CV로 epoch 수를 이미 정했다는 전제).
    반환값: (model, best_val_f1 또는 None, best_epoch, train_loss_history)
    """
    model.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_f1, best_state, best_epoch, no_improve = -1.0, None, epochs, 0
    train_loss_history = []

    for epoch in range(1, epochs + 1):
        model.train()
        rng = np.random.RandomState(epoch)
        order = rng.permutation(len(train_bags))

        total_loss = 0.0
        for idx in order:
            bag = train_bags[idx]
            x = bag.instances.to(device).float()
            target = torch.tensor([bag.true_label], device=device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_bags)
        train_loss_history.append(avg_loss)

        if val_bags is not None:
            val_acc, val_f1, _, _, _ = evaluate(model, val_bags, device)
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                best_epoch = epoch
                no_improve = 0
            else:
                no_improve += 1

            if verbose and (epoch % 5 == 0 or epoch == 1):
                log(f"    epoch {epoch:3d}/{epochs} | train_loss {avg_loss:.4f} "
                    f"| val_acc {val_acc:.3f} | val_f1_macro {val_f1:.3f} "
                    f"| best_f1 {best_val_f1:.3f}")

            if use_early_stopping and no_improve >= patience:
                if verbose:
                    log(f"    early stopping @ epoch {epoch} (patience={patience})")
                break
        else:
            if verbose and (epoch % 5 == 0 or epoch == 1):
                log(f"    epoch {epoch:3d}/{epochs} | train_loss {avg_loss:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, (best_val_f1 if val_bags is not None else None), best_epoch, train_loss_history


def compute_auc_macro(y_true, y_proba, n_classes):
    from sklearn.metrics import roc_auc_score
    present = set(y_true.tolist())
    if len(present) < 2:
        return float("nan")
    try:
        return float(roc_auc_score(
            y_true, y_proba, multi_class="ovr", average="macro",
            labels=list(range(n_classes)),
        ))
    except ValueError:
        return float("nan")


def compute_auc_per_class(y_true, y_proba, n_classes):
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import label_binarize
    y_bin = label_binarize(y_true, classes=list(range(n_classes)))
    aucs = {}
    for c in range(n_classes):
        col_sum = y_bin[:, c].sum()
        if col_sum == 0 or col_sum == len(y_bin):
            aucs[c] = float("nan")
        else:
            aucs[c] = float(roc_auc_score(y_bin[:, c], y_proba[:, c]))
    return aucs


def compute_roc_curve_points(y_true, y_proba, n_classes, class_names):
    from sklearn.metrics import roc_curve
    from sklearn.preprocessing import label_binarize
    import pandas as pd

    y_bin = label_binarize(y_true, classes=list(range(n_classes)))
    rows = []
    for c in range(n_classes):
        col_sum = y_bin[:, c].sum()
        if col_sum == 0 or col_sum == len(y_bin):
            continue
        fpr, tpr, thr = roc_curve(y_bin[:, c], y_proba[:, c])
        for f, t, th in zip(fpr, tpr, thr):
            rows.append({"class": class_names[c], "fpr": f, "tpr": t, "threshold": th})
    return pd.DataFrame(rows)


def relabel_bags(bags, labels):
    return [
        BagObject(patient_id=b.patient_id, instances=b.instances, true_label=int(l))
        for b, l in zip(bags, labels)
    ]
