# !/bin/bash
export TRANSFORMERS_OFFLINE=1

CUDA_VISIBLE_DEVICES=0
gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}
SPLIT="things"

CKPT="InternVL2_5-78B"
IDX=0
ANS_MODE=1

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m test_intern2_5vl \
    --model-path /data/llm-weight/InternVL2_5-78B \
    --question-file ./dataset/CoP_dataset_${SPLIT}_test.json \
    --image-folder /path/to/image \
    --answers-file ./answers/$CKPT/$SPLIT/${CHUNKS}_${ANS_MODE}_${IDX}.jsonl \
    --num-chunks $CHUNKS \
    --chunk-idx 0 \
    --temperature 0 \
    --answer_first $ANS_MODE \
    --conv-mode vicuna_v1 &

wait

output_file=./answers/$CKPT/$SPLIT/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./answers/$CKPT/$SPLIT/${CHUNKS}_${ANS_MODE}_${IDX}.jsonl >> "$output_file"
done

echo $SPLIT

python eval.py \
    --result-file $output_file \
    --internvl True


# !/bin/bash
export TRANSFORMERS_OFFLINE=1

CUDA_VISIBLE_DEVICES=0
gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}
SPLIT="things"

CKPT="InternVL2_5-38B"
IDX=0
ANS_MODE=1

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m test_intern2_5vl \
    --model-path /data/llm-weight/InternVL2_5-38B \
    --question-file ./dataset_2_24/CoP_dataset_${SPLIT}_test.json \
    --image-folder /data/qiong_code/data \
    --answers-file ./answers/$CKPT/$SPLIT/${CHUNKS}_${ANS_MODE}_${IDX}.jsonl \
    --num-chunks $CHUNKS \
    --chunk-idx 0 \
    --temperature 0 \
    --answer_first $ANS_MODE \
    --conv-mode vicuna_v1 &

wait

output_file=./answers/$CKPT/$SPLIT/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./answers/$CKPT/$SPLIT/${CHUNKS}_${ANS_MODE}_${IDX}.jsonl >> "$output_file"
done

echo $SPLIT

python eval.py \
    --result-file $output_file \
    --internvl True