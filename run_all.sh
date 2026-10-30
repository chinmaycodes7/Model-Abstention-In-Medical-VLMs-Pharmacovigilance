#!/usr/bin/env bash
# ==============================================================================
# MAD-MED LLaVA-Med End-to-End Pipeline
# ==============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# Dataset paths (adjust if dataset is located elsewhere)
DATA_CSV_DIR="${SCRIPT_DIR}/../data/dataset/csv"
DATA_IMG_DIR="${SCRIPT_DIR}/../data/dataset/images/task_1_VQA"

TRAIN_CSV="${DATA_CSV_DIR}/task1_VQA_train_1000.csv"
TEST_CSV="${DATA_CSV_DIR}/task1_VQA_test_504.csv"

ADAPTER_DIR="${SCRIPT_DIR}/adapters/llava-med-madmed"
OUTPUT_DIR="${SCRIPT_DIR}/outputs"
OUTPUT_CSV="${OUTPUT_DIR}/eval_image_text.csv"
SUMMARY_CSV="${OUTPUT_DIR}/summary_image_text.csv"

# Hyperparameters matching the paper reproduction
MODE="image_text" # options: image, text, image_text
EPOCHS=3
BATCH_SIZE=1
GRAD_ACCUM=4
LR="5e-5"
LAMBDA="2.0"
GPU_ID=0

echo "=========================================================="
echo " Starting MAD-MED Pipeline"
echo " Working Dir: ${SCRIPT_DIR}"
echo " Train CSV:   ${TRAIN_CSV}"
echo " Test CSV:    ${TEST_CSV}"
echo " Images:      ${DATA_IMG_DIR}"
echo " Mode:        ${MODE}"
echo "=========================================================="

# Check if data files exist
if [ ! -f "${TRAIN_CSV}" ] || [ ! -d "${DATA_IMG_DIR}" ]; then
    echo "Warning: Default dataset files not found at ${DATA_CSV_DIR}."
    echo "Please configure TRAIN_CSV, TEST_CSV, and DATA_IMG_DIR in run_all.sh."
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"

# Step 1: Train LoRA adapter
if [ ! -d "${ADAPTER_DIR}" ]; then
    echo ""
    echo ">>> [Step 1/3] Training MAD-MED LoRA adapter..."
    python3 -u -m scripts.train_madmed_llava_med \
        --train-csv "${TRAIN_CSV}" \
        --image-root "${DATA_IMG_DIR}" \
        --output-dir "${ADAPTER_DIR}" \
        --abstention-weight "${LAMBDA}" \
        --epochs "${EPOCHS}" \
        --batch-size "${BATCH_SIZE}" \
        --grad-accum-steps "${GRAD_ACCUM}" \
        --learning-rate "${LR}" \
        --no-4bit \
        --gpu "${GPU_ID}"
else
    echo ""
    echo ">>> [Step 1/3] Adapter found at ${ADAPTER_DIR}. Skipping training."
fi

# Step 2: Run inference / evaluation
echo ""
echo ">>> [Step 2/3] Running evaluation (mode: ${MODE})..."
python3 -u -m scripts.evaluate_llava_med \
    --csv "${TEST_CSV}" \
    --image-root "${DATA_IMG_DIR}" \
    --mode "${MODE}" \
    --model finetuned \
    --adapter-path "${ADAPTER_DIR}" \
    --output-csv "${OUTPUT_CSV}" \
    --gpu "${GPU_ID}"

# Step 3: Compute abstention & generation metrics
echo ""
echo ">>> [Step 3/3] Calculating quantitative metrics..."
python3 -m scripts.evaluate_abstention \
    --csv "${OUTPUT_CSV}" \
    --device "cuda:${GPU_ID}" \
    --output-summary "${SUMMARY_CSV}"

echo ""
echo "Pipeline completed successfully!"
echo "Outputs saved to: ${OUTPUT_CSV} and ${SUMMARY_CSV}"
