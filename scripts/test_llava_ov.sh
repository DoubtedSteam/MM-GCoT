# !/bin/bash

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}
SPLIT="things"

CKPT="llava-ov-7b"

mv llava_ov llava

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m test_llava_ov \
        --model-path lmms-lab/llava-onevision-qwen2-7b-ov \
        --question-file ./dataset/CoP_dataset_${SPLIT}_test.json \
        --image-folder /path/to/image \
        --answers-file ./answers/$CKPT/$SPLIT/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --answer_first 0 \
        --conv-mode vicuna_v1 &
done

wait

mv llava llava_ov

output_file=./answers/$CKPT/$SPLIT/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./answers/$CKPT/$SPLIT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python eval.py \
    --result-file $output_file
