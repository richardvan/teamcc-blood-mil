#!/usr/bin/env python3
"""
07_5_copy_rank1_images.py — rank-1(attention 최고점) 세포 이미지를 한 폴더에 모으기

attention_rank1_all_patients.csv (07_4_attention_rank1_all_patients.py 산출물)에서
rank==1 행만 뽑아, 원본 이미지를 organized_data/{patient_id}/{filename} 에서 찾아
아래 두 폴더로 복사합니다.

  blood_mil_project/RESULTS_TOP_ATTENTION_IMAGES/
    holdout/   ← split == "holdout"    (진짜 holdout 28명)
    test/      ← split == "train_pool" (CV에서 매 fold 한 번씩 test로 쓰인 나머지 161명)

파일명 형식: {patient_id}.{이미지 번호}{확장자}
  예) cancer.CBFB_MYH11.AQK / image_145.tif → cancer.CBFB_MYH11.AQK.145.tif
  (한 폴더 안에 다 모여도 patient_id + 이미지 번호만 보고 바로 구분 가능하게)

Usage:
  cd /home/sp00001/blood_mil_project/soeun_scripts
  python 07_5_copy_rank1_images.py
"""

import argparse
import os
import re
import shutil
from pathlib import Path

import pandas as pd

from mil_common import PROJECT_DIR, MODEL_ROOT, ORGANIZED_DIR, log, update_latest_symlink


SPLIT_TO_FOLDER = {
    "holdout":    "holdout",
    "train_pool": "test",
}


def make_new_filename(patient_id: str, filename: str) -> str:
    """patient_id.이미지번호.확장자 형태로 변환. 번호를 못 찾으면 원본 파일명을 그대로 붙임."""
    ext = Path(filename).suffix  # ".tif"
    m = re.search(r"(\d+)", filename)
    if m:
        return f"{patient_id}.{m.group(1)}{ext}"
    return f"{patient_id}.{filename}"


def main():
    parser = argparse.ArgumentParser(description="rank-1 attention 세포 이미지 모으기")
    parser.add_argument("--run_id", type=str, default="latest",
                        help="어느 학습 실행 결과를 쓸지. 기본값 latest는 "
                             "가장 최근 학습(07_2)이 갱신한 심볼릭 링크를 따라감. "
                             "CSV 경로와 이미지 출력 폴더 둘 다 이 run_id 밑에 저장됨.")
    parser.add_argument("--csv", type=str, default=None,
                        help="직접 지정하고 싶으면 여기에 CSV 경로. 기본값은 "
                             "run_id 기준으로 자동 계산됨.")
    parser.add_argument("--out_root", type=str,
                        default=str(PROJECT_DIR / "RESULTS_TOP_ATTENTION_IMAGES"))
    args = parser.parse_args()

    USER_TAG_DIR = MODEL_ROOT / "gen3_attention" / "soeun"
    run_id = args.run_id

    # "latest"는 이미지 출력 폴더(RESULTS_TOP_ATTENTION_IMAGES)에서도 별도로 심볼릭 링크를
    # 관리하므로, 여기서 미리 실제 run_id 문자열로 풀어둡니다 (안 그러면 "latest"라는
    # 이름의 진짜 폴더가 먼저 생겨서 나중에 심볼릭 링크 갱신과 충돌할 수 있음).
    if run_id == "latest":
        latest_link = USER_TAG_DIR / "latest"
        if not latest_link.is_symlink():
            raise FileNotFoundError(
                f"{latest_link} 심볼릭 링크가 없습니다 — 07_2_attention_mil_train.py를 "
                f"먼저 실행해서 모델을 최소 한 번 학습해야 합니다."
            )
        run_id = os.readlink(latest_link)
        log(f"[0] run_id='latest' → 실제로는 {run_id}")

    if args.csv is not None:
        csv_path = Path(args.csv)
    else:
        csv_path = USER_TAG_DIR / run_id / "artifacts" / "attention_rank1_all_patients.csv"

    out_root = Path(args.out_root)
    run_out_dir = out_root / run_id   # 실제 이미지가 저장되는 실제 폴더 (run_id별로 분리)

    log(f"[0] run_id = {run_id}")
    log(f"[1] CSV 로드 중 → {csv_path}")
    df = pd.read_csv(csv_path)
    df = df[df["rank"] == 1].copy()
    log(f"[1] rank==1 행 {len(df)}개 (환자 수와 같아야 정상)")

    for folder_name in SPLIT_TO_FOLDER.values():
        (run_out_dir / folder_name).mkdir(parents=True, exist_ok=True)

    n_ok, n_missing = 0, 0
    missing_list = []

    for _, row in df.iterrows():
        patient_id = row["patient_id"]
        filename   = row["filename"]
        split      = row["split"]

        dest_folder = SPLIT_TO_FOLDER.get(split)
        if dest_folder is None:
            log(f"  [WARN] {patient_id}: 알 수 없는 split '{split}' — 건너뜀")
            continue

        src_path = ORGANIZED_DIR / patient_id / filename
        if not src_path.exists():
            n_missing += 1
            missing_list.append(str(src_path))
            continue

        new_name = make_new_filename(patient_id, filename)
        dst_path = run_out_dir / dest_folder / new_name
        shutil.copy2(src_path, dst_path)
        n_ok += 1

    log(f"[2] 복사 완료 — 성공 {n_ok}개, 원본 없음 {n_missing}개")
    if missing_list:
        log("  누락된 원본 파일 목록:")
        for p in missing_list:
            log(f"    {p}")

    for split, folder_name in SPLIT_TO_FOLDER.items():
        n = len(df[df["split"] == split])
        log(f"  {folder_name:<10} ({split}): {n}개 대상 → {run_out_dir / folder_name}")

    update_latest_symlink(out_root, run_id)
    log(f"  최신 결과는 항상 → {out_root / 'latest'}")
    log("DONE")


if __name__ == "__main__":
    main()
