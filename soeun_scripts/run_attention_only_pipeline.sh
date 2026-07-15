#!/bin/bash
# run_attention_only_pipeline.sh — Attention 계열만 실행 (CNN 제외)
#
# CNN-MIL(06_2, 06_3)은 팀원이 따로 돌리고 있어서 뺐습니다.
# Attention-MIL(gated, 07_2~07_5) + Class-wise Attention-MIL(SCEMILA 방식, 07_6~07_7)만 실행합니다.

set -e

RUN_ID=$(date +run_%Y%m%d_%H%M%S)
echo "=================================================="
echo " RUN_ID = ${RUN_ID}"
echo " 이 실행의 모든 결과는 이 RUN_ID 하위 폴더에 저장됩니다."
echo "=================================================="

EPOCHS=${EPOCHS:-60}

echo ""
echo "########## [1/6] Attention-MIL 학습 ##########"
python 07_2_attention_mil_train.py --run_id "$RUN_ID" --epochs "$EPOCHS"

echo ""
echo "########## [2/6] Attention-MIL holdout 평가 ##########"
python 07_3_attention_mil_holdout_eval.py --run_id "$RUN_ID"

echo ""
echo "########## [3/6] Attention-MIL rank-1 세포 추출 (전체 환자) ##########"
python 07_4_attention_rank1_all_patients.py --run_id "$RUN_ID"

echo ""
echo "########## [4/6] rank-1 이미지 파일 복사 ##########"
python 07_5_copy_rank1_images.py --run_id "$RUN_ID"

echo ""
echo "########## [5/6] Class-wise Attention MIL 학습 (SCEMILA 방식) ##########"
python 07_6_multi_attention_mil_train.py --run_id "$RUN_ID" --epochs "$EPOCHS"

echo ""
echo "########## [6/6] Class-wise Attention MIL holdout 평가 ##########"
python 07_7_multi_attention_mil_holdout_eval.py --run_id "$RUN_ID"

echo ""
echo "=================================================="
echo " Attention 파이프라인 완료. RUN_ID = ${RUN_ID}"
echo ""
echo " 결과 위치 (latest 심볼릭 링크가 이번 실행을 가리킴):"
echo "   models/gen3_attention/soeun/latest/"
echo "   models/gen3_attention_classwise/soeun/latest/"
echo "   RESULTS_TOP_ATTENTION_IMAGES/latest/"
echo "=================================================="
