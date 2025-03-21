# !/bin/bash
export TRANSFORMERS_OFFLINE=1

CUDA_VISIBLE_DEVICES=0
gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}
SPLIT="things"

CKPT="qwen2_5-72B"
IDX=0

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m test_qwen2_5vl \
    --model-path /data/llm-weight/Qwen2.5-VL-72B-Instruct \
    --question-file ./dataset/CoP_dataset_${SPLIT}_test.json \
    --image-folder /path/to/image \
    --answers-file ./answers/$CKPT/$SPLIT/${CHUNKS}_${IDX}_bbox.jsonl \
    --num-chunks $CHUNKS \
    --chunk-idx $IDX \
    --temperature 0 \
    --answer_first 0 \
    --conv-mode vicuna_v1 &

wait

output_file=./answers/$CKPT/$SPLIT/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./answers/$CKPT/$SPLIT/${CHUNKS}_${IDX}_bbox.jsonl >> "$output_file"
done

python eval.py \
    --result-file $output_file \
    --qwen True