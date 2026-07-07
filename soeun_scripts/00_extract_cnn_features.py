#!/usr/bin/env python3
"""
00_extract_cnn_features.py — 환자별 인스턴스(cell) 임베딩 추출 및 캐싱

CNN-MIL, Attention-MIL 이 공유하는 전처리 단계입니다.
Frozen ResNet50(ImageNet 사전학습)으로 .tif 이미지 한 장당 2048차원 벡터를 뽑고,
환자(폴더) 단위로 (N_instance, 2048) 텐서를 cache/cnn_features/{folder}.pt 에 저장합니다.

왜 따로 분리했나:
  - 환자당 이미지가 수백~수천 장 → CNN forward pass가 가장 비싼 연산.
  - CNN-MIL / Attention-MIL 학습 중 매 epoch마다 다시 계산하면 GPU 시간 낭비.
  - 한 번 뽑아 캐싱해두면 이후 두 모델 모두 "임베딩 → pooling → classifier"만
    빠르게 반복 학습할 수 있음 (분류기 자체는 가벼움).
  - 나중에 backbone을 fine-tune하고 싶으면 --finetune_backbone 옵션으로
    이 스크립트를 건너뛰고 CNN-MIL/Attention-MIL 안에서 end-to-end로 돌릴 수 있음
    (각 스크립트의 --finetune 옵션 참고).

Usage:
  cd /home/sp00001/blood_mil_project/soeun_scripts
  python 00_extract_cnn_features.py                  # 전체 환자
  python 00_extract_cnn_features.py --backbone resnet50
"""

import argparse
import time

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image

from mil_common import (
    ORGANIZED_DIR, FEAT_CACHE_DIR, list_patient_dirs, log,
)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

BACKBONES = {
    "resnet18": (models.resnet18, models.ResNet18_Weights.IMAGENET1K_V1, 512),
    "resnet50": (models.resnet50, models.ResNet50_Weights.IMAGENET1K_V2, 2048),
}


class CellDataset(Dataset):
    """한 환자 폴더 안의 .tif 이미지들을 로드하는 Dataset."""

    def __init__(self, tif_paths, transform):
        self.paths = tif_paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        return self.transform(img)


def build_backbone(name: str, device: str):
    ctor, weights, feat_dim = BACKBONES[name]
    backbone = ctor(weights=weights)
    backbone.fc = nn.Identity()
    backbone.eval().to(device)
    for p in backbone.parameters():
        p.requires_grad = False
    return backbone, feat_dim


@torch.no_grad()
def extract_patient_embedding(patient_dir, backbone, transform, device,
                               batch_size: int = 64, num_workers: int = 4):
    tif_files = sorted(patient_dir.glob("*.tif"))
    if not tif_files:
        raise ValueError(f"No .tif images in {patient_dir}")

    ds = CellDataset(tif_files, transform)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                         num_workers=num_workers, pin_memory=(device == "cuda"))

    feats = []
    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        out = backbone(batch).cpu()
        feats.append(out)
    return torch.cat(feats, dim=0)   # (N_instance, feat_dim)


def main():
    parser = argparse.ArgumentParser(description="CNN 인스턴스 임베딩 추출 & 캐싱")
    parser.add_argument("--backbone", choices=list(BACKBONES.keys()), default="resnet50")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true",
                         help="이미 캐시된 환자도 다시 추출")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"device: {device}")
    if device == "cpu":
        log("[WARN] GPU를 찾지 못했습니다. CPU로는 매우 느립니다 — GPU 서버에서 실행하세요.")

    FEAT_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    backbone, feat_dim = build_backbone(args.backbone, device)
    log(f"Backbone: {args.backbone} (frozen), feature_dim={feat_dim}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    patient_dirs = list_patient_dirs(ORGANIZED_DIR)
    log(f"총 {len(patient_dirs)}명 환자 발견")

    t0 = time.time()
    n_done, n_skipped = 0, 0
    for i, pdir in enumerate(patient_dirs):
        out_path = FEAT_CACHE_DIR / f"{pdir.name}.pt"
        if out_path.exists() and not args.overwrite:
            n_skipped += 1
            continue

        emb = extract_patient_embedding(
            pdir, backbone, transform, device,
            batch_size=args.batch_size, num_workers=args.num_workers,
        )
        torch.save(emb, out_path)
        n_done += 1

        if (i + 1) % 5 == 0 or (i + 1) == len(patient_dirs):
            log(f"  [{i+1}/{len(patient_dirs)}] {pdir.name} "
                f"→ {tuple(emb.shape)}  ({time.time()-t0:.1f}s elapsed, "
                f"{n_done} 신규 / {n_skipped} 스킵)")

    log(f"완료: 신규 추출 {n_done}명, 이미 캐시되어 스킵 {n_skipped}명")
    log(f"캐시 위치: {FEAT_CACHE_DIR}")


if __name__ == "__main__":
    main()
