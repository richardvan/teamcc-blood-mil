#!/usr/bin/env python3
"""
07_4_attention_rank1_all_patients.py — 전체 환자(189명)의 rank-1 attention 세포 추출

attention_top_cells_holdout.csv 는 holdout 28명만 다뤘는데, 이 스크립트는
metadata_for_multiclass.csv 에 있는 전체 환자(control 포함, train+holdout 다)에 대해
"이 환자에서 attention이 가장 높았던 세포 1장"을 뽑습니다.

★ 반드시 같이 봐야 할 것: `split` 컬럼
    - "holdout" : 모델이 학습에서 한 번도 본 적 없는 환자 → 여기서 rank-1이 말이 되면
                  모델이 실제로 일반화 가능한 패턴을 학습했다는 근거로 볼 수 있음
    - "train_pool" : 모델이 라벨을 보고 학습한 환자 → rank-1이 그럴듯해 보여도
                  overfitting에 의한 결과일 수 있어 해석에 주의 필요

★ attn_weight_vs_uniform 컬럼
    원본 attn_weight는 그 환자의 세포 수(N)에 따라 스케일이 다름 (softmax가 N개에 걸쳐
    정규화되므로 평균이 1/N). 그래서 "균등 분배(1/N) 대비 몇 배 더 집중했는가"를 같이
    계산해 넣었습니다. 환자 간 비교는 원본 attn_weight보다 이 값으로 하는 게 공정합니다.

전제 조건:
  - 07_2_attention_mil_train.py 로 최종 모델(attention_mil_v1.pt)이 저장되어 있어야 함
  - cache/cnn_features/*.pt 에 전체 환자 임베딩이 캐싱되어 있어야 함

Usage:
  cd /home/sp00001/blood_mil_project/soeun_scripts
  python 07_4_attention_rank1_all_patients.py
"""

import argparse

import numpy as np
import pandas as pd
import torch

from mil_common import (
    PROJECT_DIR, MODEL_ROOT, CLASS_NAMES,
    log, build_bag_objects, get_instance_filenames,
    load_metadata, get_holdout_folders_from_metadata,
)
from attention_mil_common import AttentionMIL, AttentionMILWrapper


def main():
    parser = argparse.ArgumentParser(description="전체 환자 rank-1 attention 세포 추출")
    parser.add_argument("--metadata_file", type=str, default="metadata_for_multiclass.csv")
    parser.add_argument("--model_name", type=str, default="attention_mil_v1")
    parser.add_argument("--top_k", type=int, default=1,
                        help="환자당 저장할 top-k 개수 (기본 1 = rank-1만)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_gen = "gen3_attention"
    save_dir = MODEL_ROOT / model_gen / "artifacts"
    save_dir.mkdir(parents=True, exist_ok=True)

    log("=" * 55)
    log("START: 전체 환자 rank-1 attention 세포 추출")
    log(f"  Device : {device}")
    log("=" * 55)

    # ── 모델 로드 (재학습 없음, 이미 저장된 최종 모델 사용) ─────
    model_path = MODEL_ROOT / model_gen / f"{args.model_name}.pt"
    log(f"[1] 모델 로드 중 → {model_path}")
    ckpt = torch.load(model_path, map_location=device)
    model = AttentionMIL(
        in_dim=ckpt["in_dim"], hidden_dim=ckpt["hidden_dim"], attn_dim=ckpt["attn_dim"],
        n_classes=ckpt["n_classes"], dropout=ckpt["dropout"], gated=ckpt["gated"],
    )
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    wrapper = AttentionMILWrapper(model, device)
    log("[1] 완료")

    # ── 전체 환자 목록 (control 포함, train+holdout 다) ─────────
    log(f"[2] {args.metadata_file} 로드 중...")
    meta = load_metadata(PROJECT_DIR / args.metadata_file)
    holdout_folders = get_holdout_folders_from_metadata(meta)
    all_folders = meta["folder"].tolist()
    log(f"[2] 완료 — 전체 {len(all_folders)}명 "
        f"(train_pool {len(all_folders) - len(holdout_folders)}명 / holdout {len(holdout_folders)}명)")

    log("[3] 임베딩 로드 중...")
    bags = build_bag_objects(all_folders)
    log(f"[3] 완료 — {len(bags)}명")

    # ── 환자별 rank-1(또는 top-k) attention 세포 추출 ──────────
    log("[4] attention 계산 중...")
    rows = []
    for i, bag in enumerate(bags):
        result = wrapper.predict_bag_with_attention(bag.instances)
        attn = result["attn_weights"]
        n_instances = len(attn)
        filenames = get_instance_filenames(bag.patient_id)

        if len(filenames) != n_instances:
            log(f"  [WARN] {bag.patient_id}: 파일명 수({len(filenames)}) != "
                f"인스턴스 수({n_instances}) — 건너뜀")
            continue

        uniform_weight = 1.0 / n_instances
        top_idx = np.argsort(-attn)[:args.top_k]

        for rank, idx in enumerate(top_idx, start=1):
            rows.append({
                "patient_id":            bag.patient_id,
                "split":                 "holdout" if bag.patient_id in holdout_folders else "train_pool",
                "true_subtype":          CLASS_NAMES[bag.true_label],
                "pred_subtype":          CLASS_NAMES[result["pred_label"]],
                "correct":               int(bag.true_label == result["pred_label"]),
                "n_instances":           n_instances,
                "rank":                  rank,
                "filename":              filenames[idx],
                "attn_weight":           float(attn[idx]),
                "attn_weight_vs_uniform": float(attn[idx] / uniform_weight),
            })

        if (i + 1) % 20 == 0 or (i + 1) == len(bags):
            log(f"  [{i+1}/{len(bags)}] 처리 완료")

    df = pd.DataFrame(rows)
    out_path = save_dir / "attention_rank1_all_patients.csv"
    df.to_csv(out_path, index=False)
    log(f"[4] 완료 — 저장 → {out_path}")

    # ── 간단 요약 (split별, 정답 여부별 rank-1 집중도) ──────────
    if args.top_k >= 1:
        rank1 = df[df["rank"] == 1]
        log("[5] rank-1 집중도 요약 (attn_weight_vs_uniform, 클수록 특정 세포에 쏠렸다는 뜻):")
        for split_name, g in rank1.groupby("split"):
            log(f"  {split_name:<12}: mean={g['attn_weight_vs_uniform'].mean():.1f}배 "
                f"| median={g['attn_weight_vs_uniform'].median():.1f}배 "
                f"| n={len(g)}")

    log("=" * 55)
    log("DONE")
    log(f"  결과: {out_path}")
    log("=" * 55)


if __name__ == "__main__":
    main()
