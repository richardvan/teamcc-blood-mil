#!/usr/bin/env python3
"""
multi_attention_mil_common.py — Class-wise Attention MIL (SCEMILA 논문 방식)

07_2/07_3(gated attention, 클래스 공유)과의 차이는 딱 하나: attention을
클래스마다 독립적으로 계산합니다 (Ilse et al. gated attention → SCEMILA류
class-wise attention). 논문 수식 그대로:

    α_{i,k} = softmax_k{ w_i^T tanh(V_i h_k) },  ∀ class i
    z_i     = Σ_k α_{i,k} h_k                      (클래스마다 다른 bag 벡터)
    y_i     = f_cls(z_i; ρ)                        (classifier는 클래스 간 공유)

학습/평가 루프(train_model, evaluate, AUC/ROC 계산 등)는 attention_mil_common.py
것을 그대로 재사용합니다 — forward()가 (logits, attn)만 돌려주면 되므로
모델 구조만 바뀌면 됩니다.
"""

import numpy as np
import torch
import torch.nn as nn

# 학습 루프/지표 계산은 gated-attention 버전과 완전히 동일하므로 재사용
from attention_mil_common import (
    evaluate, train_model, relabel_bags,
    compute_auc_macro, compute_auc_per_class, compute_roc_curve_points,
)


class ClassWiseAttentionMIL(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, attn_dim: int, n_classes: int,
                 dropout: float = 0.3):
        super().__init__()
        self.n_classes = n_classes
        self.attn_dim = attn_dim

        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        # V_i 를 클래스별로 따로 두는 대신, (hidden -> n_classes*attn_dim) 한 번에 계산 후
        # (N, n_classes, attn_dim)로 reshape — 수학적으로 클래스별 V_i를 쌓은 것과 동일
        self.V = nn.Linear(hidden_dim, n_classes * attn_dim, bias=False)
        self.w = nn.Parameter(torch.randn(n_classes, attn_dim) * 0.01)  # w_i, 클래스별

        self.classifier = nn.Linear(hidden_dim, 1)  # f_cls, 클래스 간 공유

    def encode_and_attend(self, bag: torch.Tensor):
        """
        h: (N, hidden) 인스턴스별 인코딩
        z: (n_classes, hidden) 클래스별 bag 벡터
        attn: (N, n_classes) 클래스별 attention 가중치 (각 열의 합 = 1)
        """
        h = self.encoder(bag)                                   # (N, hidden)
        N = h.shape[0]

        v = torch.tanh(self.V(h)).view(N, self.n_classes, self.attn_dim)  # (N, C, A)
        scores = torch.einsum("nca,ca->nc", v, self.w)          # (N, C)
        attn = torch.softmax(scores, dim=0)                      # 인스턴스 축으로 정규화, (N, C)
        z = torch.einsum("nc,nh->ch", attn, h)                   # (C, hidden)
        return h, z, attn

    def forward(self, bag: torch.Tensor):
        _, z, attn = self.encode_and_attend(bag)
        logits = self.classifier(z).squeeze(-1).unsqueeze(0)     # (1, n_classes)
        return logits, attn


class ClassWiseAttentionMILWrapper:
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
    def predict_bag_proba(self, bag_instances) -> np.ndarray:
        x = bag_instances.to(self.device)
        if x.dtype != torch.float32:
            x = x.float()
        logits, _ = self.model(x)
        return torch.softmax(logits, dim=1)[0].cpu().numpy()

    @torch.no_grad()
    def predict_bag_with_attention(self, bag_instances):
        """
        예측된 클래스에 대한 attention 가중치를 반환합니다
        (그 판단을 내릴 때 어떤 세포를 봤는지 — SCEMILA Fig.2/3과 동일한 해석).
        """
        x = bag_instances.to(self.device)
        if x.dtype != torch.float32:
            x = x.float()
        logits, attn = self.model(x)                             # attn: (N, n_classes)
        probs = torch.softmax(logits, dim=1)[0]
        pred_label = int(torch.argmax(probs).item())
        pred_score = float(probs[pred_label].item())
        attn_for_pred = attn[:, pred_label].cpu().numpy()          # (N,) — 예측 클래스 기준
        return {
            "pred_score": pred_score,
            "pred_label": pred_label,
            "attn_weights": attn_for_pred,
            "attn_weights_all_classes": attn.cpu().numpy(),        # (N, n_classes) 전체 보존
        }

    @torch.no_grad()
    def get_bag_vector(self, bag_instances) -> np.ndarray:
        """
        클래스별 bag 벡터(z_1..z_C)를 이어붙인 것 — PCA 시각화용.
        (gated attention은 클래스당 z가 하나뿐이라 바로 쓰지만, 여기는 클래스마다
         z가 달라서 전부 이어붙여 하나의 "환자 표현 벡터"로 씀)
        """
        x = bag_instances.to(self.device)
        if x.dtype != torch.float32:
            x = x.float()
        _, z, _ = self.model.encode_and_attend(x)
        return z.cpu().numpy().flatten()   # (n_classes * hidden,)
