import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

from PIL import Image
import math


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def eval_model(args):
    # Model
    model_name = args.model_path.split('/')[-1]
    print(args.model_path)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(args.model_path)
    
    # Data
    with open(args.question_file, "r") as fa:
        questions = json.load(fa)
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    
    for line in tqdm(questions):
        idx = line["id"]
        image_file = line["img"]
        
        if args.answer_first:
            first_turn = line["question"] + "\nAnswer the question using a single word or phrase."
        else:
            first_turn = line["question"] + "\nOutline the position of the final answer and output the coordinate in JSON format."
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": "file://"+ os.path.join(args.image_folder, image_file),
                    },
                    {"type": "text", "text": first_turn},
                ],
            }
        ]
        # Preparation for inference
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        # Inference: Generation of the output
        generated_ids = model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        ori_outputs = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        if args.answer_first:
            ans_outputs = ori_outputs
        else:
            bbox_outputs = ori_outputs
        
        
        if args.answer_first:
            second_turn = "Outline the position of the final answer and output the coordinate in JSON format."
        else:
            second_turn = "Answer the question using a single word or phrase."
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": "file://"+ os.path.join(args.image_folder, image_file),
                    },
                    {"type": "text", "text": first_turn},
                ],
            },
            {
                "role": "gpt",
                "content": [
                    {"type": "text", "text": ori_outputs},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": second_turn},
                ],
            },
        ]
        # Preparation for inference
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")
        
        # Inference: Generation of the output
        generated_ids = model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        ori_outputs = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        if args.answer_first:
            bbox_outputs = ori_outputs
        else:
            ans_outputs = ori_outputs
        
        

        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "img": image_file,
                                   "prompt": first_turn,
                                   "text": ans_outputs[0],
                                   "coor": bbox_outputs[0],
                                   "gt_text": line['answer'],
                                   "gt_coor": line['coordinate'],
                                   "answer_id": ans_id,
                                   "model_id": model_name,
                                   "metadata": {}}) + "\n")
        ans_file.flush()
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--answer_first", type=int, default=1)
    args = parser.parse_args()

    eval_model(args)
