#!/bin/bash


if [ "$#" -lt 3 ]; then
    echo "Usage: bash run_cos.sh test.mp4 ./results roughness metallic etc"
    exit 1
fi

INPUT_VIDEO=$1
FINAL_OUT_DIR=$2
MODALITIES="${@:3}"


TEMP_ID=$RANDOM
TMP_WORK_DIR="./tmp_work_$TEMP_ID"
TMP_VIDEO_IN="$TMP_WORK_DIR/video_in"
TMP_FRAMES_OUT="$TMP_WORK_DIR/frames_extracted"
TMP_INFERENCE_OUT="$TMP_WORK_DIR/inference_results"
mkdir -p "$TMP_VIDEO_IN" "$TMP_FRAMES_OUT" "$TMP_INFERENCE_OUT" "$FINAL_OUT_DIR"
VIDEO_NAME=$(basename "$INPUT_VIDEO")
VIDEO_BASENAME="${VIDEO_NAME%.*}"
ln -s "$(realpath "$INPUT_VIDEO")" "$TMP_VIDEO_IN/$VIDEO_NAME"

CHECKPOINT="checkpoints"
MODEL_DIR="Diffusion_Renderer_Inverse_Cosmos_7B"


PYTHONPATH=$(pwd) python scripts/dataproc_extract_frames_from_video.py \
    --input_folder "$TMP_VIDEO_IN" \
    --output_folder "$TMP_FRAMES_OUT" \
    --frame_rate 24 \
    --resize 448x256 \
    --max_frames 57

echo "2nd step"

CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python cosmos_predict1/diffusion/inference/inference_inverse_renderer.py \
    --checkpoint_dir checkpoints \
    --diffusion_transformer_dir Diffusion_Renderer_Inverse_Cosmos_7B \
    --dataset_path "$EXTRACTED_PATH" \
    --video_save_folder "$TMP_INFERENCE_OUT" \
    --num_video_frames 57 \
    --group_mode folder \
    --inference_passes $MODALITIES \
    --num_steps 10 \
    --resize_resolution 256 448

echo ">>> Step 3: Analyzing..."
find "$TMP_INFERENCE_OUT" -name "*.mp4" -exec cp {} "$FINAL_OUT_DIR/" \;

rm -rf "$TMP_WORK_DIR"
echo "Done, output dir: $FINAL_OUTPUT_DIR"