# !/bin/bash

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}
SPLIT="judge"

CKPT="llava-v1.5-13b-GCoT-output"

mv llava_ori llava

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m test_llava_cop \
        --model-path /mnt/82_store/wq/CoP/LLaVA-CoP/checkpoints/llava-v1.5-13b-VCoT \
        --question-file ./dataset/CoP_dataset_${SPLIT}_test.json \
        --image-folder /path/to/image \
        --answers-file ./answers/$CKPT/$SPLIT/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --cot True \
        --temperature 0 \
        --conv-mode vicuna_v1 &
done

wait

mv llava llava_ori

output_file=./answers/$CKPT/$SPLIT/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./answers/$CKPT/$SPLIT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python eval.py \
    --result-file $output_file
