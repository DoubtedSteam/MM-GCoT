import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration

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
    # disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    # model_name = get_model_name_from_path(model_path)
    model_name = model_path.split('/')[-1]
    # tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)
    model = LlavaOnevisionForConditionalGeneration.from_pretrained(
        model_path, 
        torch_dtype=torch.float16, 
        low_cpu_mem_usage=True, 
        device_map="auto"
    ).eval()
    processor = AutoProcessor.from_pretrained(model_path)
    
    # Data
    with open(args.question_file, "r") as fa:
        questions = json.load(fa)
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    
    for line in tqdm(questions):
        idx = line["id"]
        if args.answer_first:
            first_turn = line["question"] + "\nAnswer the question using a single word or phrase."
        else:
            first_turn = line["question"] + "\nPlease provide the bounding box coordinate of the region for the final answer."

        ###### For Answer ######
        conversation = [
            {
            "role": "user",
            "content": [
                {"type": "text", "text": first_turn},
                {"type": "image"},
                ],
            },
        ]
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        raw_image = Image.open(os.path.join(args.image_folder, line["img"]))
        inputs = processor(images=raw_image, text=prompt, return_tensors='pt').to(0, torch.float16)
        
        outputs = model.generate(**inputs, max_new_tokens=200, do_sample=False)
        outputs = processor.decode(outputs[0][2:], skip_special_tokens=True).split('assistant\n')[-1]
        if args.answer_first:
            ans_outputs = outputs.strip()
        else:
            bbox_outputs = outputs.strip()

        ###### For BBox ######
        if args.answer_first:
            seconde_turn = "Please provide the bounding box coordinate of the region for the final answer."
        else:
            seconde_turn = "Answer the question using a single word or phrase."
        
        conversation = [
            {
            "role": "user",
            "content": [
                {"type": "text", "text": first_turn},
                {"type": "image"},
                ],
            },
            {
            "role": "assistant",
            "content": [
                {"type": "text", "text": outputs},
                ],
            },
            {
            "role": "user",
            "content": [
                {"type": "text", "text": seconde_turn},
                ],
            },
        ]
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = processor(images=raw_image, text=prompt, return_tensors='pt').to(0, torch.float16)
        
        outputs = model.generate(**inputs, max_new_tokens=200, do_sample=False)
        outputs = processor.decode(outputs[0][2:], skip_special_tokens=True).split('assistant\n')[-1]
        if args.answer_first:
            bbox_outputs = outputs.strip()
        else:
            ans_outputs = outputs.strip()
        
        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": first_turn,
                                   "text": ans_outputs,
                                   "coor": bbox_outputs,
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
    parser.add_argument("--extra-prompt", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--test_size", type=int, default=10000000)
    parser.add_argument("--answer_first", type=int, default=1)
    args = parser.parse_args()

    eval_model(args)