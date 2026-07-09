#!/usr/bin/env python3
"""
attention_mil_common.py — Attention-MIL 모델 정의 + 학습/평가 유틸

07_2_attention_mil_train.py / 07_3_attention_mil_holdout_eval.py 가 공유합니다.
(팀원 SVM 코드처럼 train/eval 스크립트를 분리하되, 모델 정의가 두 파일에
중복되면 나중에 한쪽만 고치는 실수가 나기 쉬워서 이 부분만 공용 모듈로 뺐습니다.)
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, accuracy_score

from mil_common import log, BagObject, compute_class_weights


# ──────────────────────────────────────────────────────────────
# 모델 정의 (Gated Attention MIL, Ilse et al. 2018)
# ──────────────────────────────────────────────────────────────

class AttentionMIL(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, attn_dim: int, n_classes: int,
                 dropout: float = 0.3, gated: bool = True):
        super().__init__()
        self.gated = gated

        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.attn_V = nn.Sequential(nn.Linear(hidden_dim, attn_dim), nn.Tanh())
        if gated:
            self.attn_U = nn.Sequential(nn.Linear(hidden_dim, attn_dim), nn.Sigmoid())
        self.attn_w = nn.Linear(attn_dim, 1)

        self.classifier = nn.Linear(hidden_dim, n_classes)

    def encode_and_attend(self, bag: torch.Tensor):
        """h(인스턴스별 인코딩), z(attention-pooled bag 벡터), attn_weights 반환."""
        h = self.encoder(bag)                          # (N, hidden)

        a_v = self.attn_V(h)                           # (N, attn_dim)
        if self.gated:
            a_u = self.attn_U(h)
            scores = self.attn_w(a_v * a_u)             # (N, 1)
        else:
            scores = self.attn_w(a_v)

        attn_weights = torch.softmax(scores, dim=0)      # (N, 1)
        z = (attn_weights * h).sum(dim=0, keepdim=True)  # (1, hidden)
        return h, z, attn_weights.squeeze(1)

    def forward(self, bag: torch.Tensor):
        _, z, attn_weights = self.encode_and_attend(bag)
        logits = self.classifier(z)                      # (1, n_classes)
        return logits, attn_weights


class AttentionMILWrapper:
    """shared_functions_V2 이 요구하는 predict_bag() 인터페이스."""
    def __init__(self, model: nn.Module, device: str):
        self.model = model.eval()
        self.device = device

    @torch.no_grad()
    def predict_bag(self, bag_instances) -> dict:
        x = bag_instances.to(self.device)
        if x.dtype != torch.float32:
            x = x.float()
        logits, _ = self.model(x)
        probs = torch.softmax(logits, dim=1)[0]
        pred_label = int(torch.argmax(probs).item())
        pred_score = float(probs[pred_label].item())
        return {"pred_score": pred_score, "pred_label": pred_label}

    @torch.no_grad()
    def predict_bag_with_attention(self, bag_instances):
        x = bag_instances.to(self.device)
        if x.dtype != torch.float32:
            x = x.float()
        logits, attn = self.model(x)
        probs = torch.softmax(logits, dim=1)[0]
        pred_label = int(torch.argmax(probs).item())
        pred_score = float(probs[pred_label].item())
        return {
            "pred_score": pred_score,
            "pred_label": pred_label,
            "attn_weights": attn.cpu().numpy(),
        }

    @torch.no_grad()
    def predict_bag_proba(self, bag_instances) -> np.ndarray:
        """클래스별 확률 전체 반환 (n_classes,) — AUC 계산용."""
        x = bag_instances.to(self.device)
        if x.dtype != torch.float32:
            x = x.float()
        logits, _ = self.model(x)
        probs = torch.softmax(logits, dim=1)[0]
        return probs.cpu().numpy()

    @torch.no_grad()
    def get_bag_vector(self, bag_instances) -> np.ndarray:
        """
        attention-pooled bag 벡터 z (분류기 직전의 hidden 표현).
        SVM 코드의 "환자당 mean-pooled ResNet 벡터"에 대응하는,
        Attention-MIL 버전의 "환자당 표현 벡터" — PCA 시각화용.
        """
        x = bag_instances.to(self.device)
        if x.dtype != torch.float32:
            x = x.float()
        _, z, _ = self.model.encode_and_attend(x)
        return z.cpu().numpy().flatten()


# ──────────────────────────────────────────────────────────────
# 학습/평가 루프
# ──────────────────────────────────────────────────────────────

def evaluate(model, bags, device):
    model.eval()
    y_true, y_pred, y_proba = [], [], []
    with torch.no_grad():
        for bag in bags:
            x = bag.instances.to(device).float()
            logits, _ = model(x)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
            pred = int(np.argmax(probs))
            y_true.append(bag.true_label)
            y_pred.append(pred)
            y_proba.append(probs)
    y_true, y_pred, y_proba = np.array(y_true), np.array(y_pred), np.array(y_proba)
    acc = accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, average="macro", zero_division=0)
    return acc, f1m, y_true, y_pred, y_proba


def compute_auc_macro(y_true, y_proba, n_classes):
    """One-vs-rest macro AUC. 표본에 없는 클래스가 있으면 계산 불가하므로 NaN 처리."""
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
    """클래스별 one-vs-rest AUC. 해당 클래스가 표본에 아예 없거나 전부인 경우 NaN."""
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
    """
    클래스별 one-vs-rest ROC curve 좌표(FPR, TPR, threshold) 계산.
    ROC curve를 실제로 '그리려면' AUC 숫자 하나가 아니라 이 좌표들이 필요합니다.
    반환: pandas DataFrame (class_name, fpr, tpr, threshold 컬럼)
    """
    from sklearn.metrics import roc_curve
    from sklearn.preprocessing import label_binarize
    import pandas as pd

    y_bin = label_binarize(y_true, classes=list(range(n_classes)))
    rows = []
    for c in range(n_classes):
        col_sum = y_bin[:, c].sum()
        if col_sum == 0 or col_sum == len(y_bin):
            continue   # 이 클래스는 양성/음성이 한쪽만 있어서 곡선을 못 그림
        fpr, tpr, thr = roc_curve(y_bin[:, c], y_proba[:, c])
        for f, t, th in zip(fpr, tpr, thr):
            rows.append({"class": class_names[c], "fpr": f, "tpr": t, "threshold": th})
    return pd.DataFrame(rows)


def train_model(model, train_bags, val_bags, device, epochs, lr, weight_decay,
                 class_weights, patience=15, verbose=True, use_early_stopping=True):
    model.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_f1, best_state, no_improve = -1.0, None, 0

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
            logits, _ = model(x)
            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        val_acc, val_f1, _, _, _ = evaluate(model, val_bags, device)
        avg_loss = total_loss / len(train_bags)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
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

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val_f1


def relabel_bags(bags, labels):
    """bags와 같은 순서의 labels 배열로 true_label만 바꾼 새 BagObject 리스트 반환.
    (robustness 체크의 label-shuffle 테스트에서 사용)"""
    return [
        BagObject(patient_id=b.patient_id, instances=b.instances, true_label=int(l))
        for b, l in zip(bags, labels)
    ]
