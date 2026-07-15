#!/bin/bash
# run_full_pipeline.sh — Attention / Class-wise Attention / CNN 전체 파이프라인을
# 한 번에, 같은 RUN_ID로 순서대로 실행합니다.
#
# 왜 필요한가:
#   - 07_2(학습)를 다시 돌리면 모델(.pt)이 새로 만들어지는데, 이걸 쓰는
#     07_3(holdout 평가), 07_4(rank-1 추출), 07_5(이미지 복사)도 그 순서대로
#     다시 돌아야 "최신 모델 기준" 결과가 됩니다. 하나라도 빠뜨리면
#     예전 모델 결과랑 최신 모델 결과가 섞여서 헷갈리게 됩니다.
#   - 이 스크립트는 하나의 RUN_ID(타임스탬프)를 만들어서 모든 단계에 동일하게
#     넘겨주고, 순서를 강제해서 그런 실수를 원천 차단합니다.
#   - 각 실행은 run_id별 폴더에 저장되고, latest 심볼릭 링크가 항상 최신을
#     가리키므로 예전 결과와 안 섞입니다.
#
# 사용법:
#   cd /home/sp00001/blood_mil_project/soeun_scripts
#   bash run_full_pipeline.sh
#   (SLURM으로 돌릴 땐 sbatch 스크립트 안에서 이 파일을 그대로 실행하면 됩니다)

set -e   # 중간에 하나라도 실패하면 즉시 중단 (그래야 최신 모델 없이 다음 단계가 안 돔)

RUN_ID=$(date +run_%Y%m%d_%H%M%S)
echo "=================================================="
echo " RUN_ID = ${RUN_ID}"
echo " 이 실행의 모든 결과는 이 RUN_ID 하위 폴더에 저장됩니다."
echo "=================================================="

EPOCHS=${EPOCHS:-60}   # 환경변수로 EPOCHS=100 bash run_full_pipeline.sh 처럼 덮어쓸 수 있음

echo ""
echo "########## [1/8] Attention-MIL 학습 ##########"
python 07_2_attention_mil_train.py --run_id "$RUN_ID" --epochs "$EPOCHS"

echo ""
echo "########## [2/8] Attention-MIL holdout 평가 ##########"
python 07_3_attention_mil_holdout_eval.py --run_id "$RUN_ID"

echo ""
echo "########## [3/8] Attention-MIL rank-1 세포 추출 (전체 환자) ##########"
python 07_4_attention_rank1_all_patients.py --run_id "$RUN_ID"

echo ""
echo "########## [4/8] rank-1 이미지 파일 복사 ##########"
python 07_5_copy_rank1_images.py --run_id "$RUN_ID"

echo ""
echo "########## [5/8] Class-wise Attention MIL 학습 (SCEMILA 방식) ##########"
python 07_6_multi_attention_mil_train.py --run_id "$RUN_ID" --epochs "$EPOCHS"

echo ""
echo "########## [6/8] Class-wise Attention MIL holdout 평가 ##########"
python 07_7_multi_attention_mil_holdout_eval.py --run_id "$RUN_ID"

echo ""
echo "########## [7/8] CNN-MIL 학습 (mean pooling) ##########"
python 06_2_cnn_mil_train.py --run_id "$RUN_ID" --pooling mean --epochs "$EPOCHS"

echo ""
echo "########## [8/8] CNN-MIL holdout 평가 ##########"
python 06_3_cnn_mil_holdout_eval.py --run_id "$RUN_ID" --pooling mean

echo ""
echo "=================================================="
echo " 전체 파이프라인 완료. RUN_ID = ${RUN_ID}"
echo ""
echo " 결과 위치 (전부 latest 심볼릭 링크가 이번 실행을 가리킴):"
echo "   models/gen3_attention/soeun/latest/"
echo "   models/gen3_attention_classwise/soeun/latest/"
echo "   models/gen2_cnn/soeun/latest/"
echo "   RESULTS_TOP_ATTENTION_IMAGES/latest/"
echo ""
echo " 이번 실행만 따로 보고 싶으면 latest 대신:"
echo "   models/gen3_attention/soeun/${RUN_ID}/"
echo "   models/gen3_attention_classwise/soeun/${RUN_ID}/"
echo "   models/gen2_cnn/soeun/${RUN_ID}/"
echo "   RESULTS_TOP_ATTENTION_IMAGES/${RUN_ID}/"
echo "=================================================="
