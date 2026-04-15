#!/bin/bash


if [ "$#" -ne 2 ]; then
    echo "bash run_cos.sh <input_vid> <output_vid>"
    exit 1
fi
VIDEO_IN=$1
VIDEO_OUT=$2
TEMP_ID=$RANDOM
TMP_IN="./tmp_in_$TEMP_ID"
TMP_OUT="./tmp_out_$TEMP_ID"
mkdir -p "$TMP_IN" "$TMP_OUT"
IN_NAME=$(basename "$VIDEO_IN")
ln -s "$(realpath "$VIDEO_IN")" "$TMP_IN/$IN_NAME"
CHECKPOINT="checkpoints"
MODEL_DIR="Diffusion_Renderer_Inverse_Cosmos_7B"

CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python cosmos_predict1/diffusion/inference/inference_inverse_renderer.py \
    --checkpoint_dir $CHECKPOINT \
    --diffusion_transformer_dir $MODEL_DIR \
    --dataset_path "$TMP_IN" \
    --video_save_folder "$TMP_OUT" \
    --num_video_frames 57 \
    --chunk_mode all \
    --overlap_n_frames 8 \
    # --inference_passes roughness metallic \
    --num_steps 10 \
    --resize_resolution 256 448
RESULT_FILE=$(find "$TMP_OUT" -name "*.mp4" | head -n 1)
if [ -f "$RESULT_FILE" ]; then
    mv "$RESULT_FILE" "$VIDEO_OUT"
    echo ">>> Success! Output saved to: $VIDEO_OUT"
else
    echo ">>> Error: Inference failed, result file not found."
fi
rm -rf "$TMP_IN" "$TMP_OUT"
echo "Done, output dir: $OUTPUT_DIR"