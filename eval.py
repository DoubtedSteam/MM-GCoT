import argparse
import torch
import os
import json
import re
from tqdm import tqdm
import shortuuid

from PIL import Image
import math


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]

def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def parse_bbox(bbox_str):
    # 将字符串解析为数值列表
    if bbox_str is None:
        return [0,0,0,0]
    coord_match = re.search(r'\[([0-9., ]+)\]', bbox_str)
    print(coord_match)
    if coord_match:
        coord_match = coord_match.group(0)
    else:
        coord_match = bbox_str
    try:
        result = [float(x) for x in coord_match[1:-1].split(',')]
        if len(result) < 4:
            return [0,0,0,0]
        return result
    except:
        return [0,0,0,0]

def calculate_iou(bbox1, bbox2):
    # 提取坐标
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2

    # 计算交集区域的坐标
    x_inter1 = max(x1, x2)
    y_inter1 = max(y1, y2)
    x_inter2 = min(x1 + w1, x2 + w2)
    y_inter2 = min(y1 + h1, y2 + h2)

    # 计算交集面积
    inter_area = max(0, x_inter2 - x_inter1) * max(0, y_inter2 - y_inter1)

    # 计算并集面积
    bbox1_area = w1 * h1
    bbox2_area = w2 * h2
    union_area = bbox1_area + bbox2_area - inter_area

    # 计算 IoU
    iou = inter_area / union_area if union_area != 0 else 0

    return iou

def count_above_margin(lst, margin=0.5):
    return sum(1 for num in lst if num > margin)

def compute_center_accuracy(box1, box2):
    """
    Compute if the center point of box 2 is within box 1.

    Parameters:
    - box1 (list of float): Bounding box [x_min, y_min, x_max, y_max].
    - box2 (list of float): Bounding box [x_min, y_min, x_max, y_max].

    Returns:
    - bool: True if the center point of box 2 is within box 1, False otherwise.
    """
    # Compute the center point of box 2
    center_x = (box2[0] + box2[2]) / 2
    center_y = (box2[1] + box2[3]) / 2

    # Check if the center point is within box 1
    return box1[0] <= center_x <= box1[2] and box1[1] <= center_y <= box1[3]

def eval_model(args):
    # Model
    results = [json.loads(q) for q in open(os.path.expanduser(args.result_file), "r")]

    correct_answer = []
    IoUs = []
    center_accuracy = []

    for line in results:
        answer = line["text"]
        gt_answer = line["gt_text"]

        answer = answer.replace('.', '').lower()
        if args.v7w:
            answer = answer[0]
        gt_answer = gt_answer.lower()
        
        if answer == gt_answer:
            correct_answer.append(1)
        else:
            correct_answer.append(0)

        bbox = parse_bbox(line["coor"])
        if args.qwen:
            image_path = line['img']
            img = Image.open("/data/qiong_code/data/" + image_path)
            width, height = img.size
            bbox = [bbox[0] / width, bbox[1] / height, (bbox[2]-bbox[0]) / width, (bbox[3]-bbox[1]) / height]
        if args.internvl:
            bbox = [bbox[0] / 1000, bbox[1] / 1000, (bbox[2]-bbox[0]) / 1000, (bbox[3]-bbox[1]) / 1000]
            
        print(bbox)
            
        gt_bbox = parse_bbox(line["gt_coor"])

        IoUs.append(calculate_iou(bbox, gt_bbox))

        if IoUs[-1] > 0.5:
            center_accuracy.append(1)
        else:
            center_accuracy.append(0)
        
    CR = 0
    CM = 0
    WR = 0
    for acc, center in zip(correct_answer, center_accuracy):
        if acc==1 and center==1:
            CR += 1
        if acc==1 and center==0:
            CM += 1
        if acc==0 and center==1:
            WR += 1

    print('Accuracy:')
    print(sum(correct_answer)/len(correct_answer)*100)
    # print("Average IoUs:")
    # print(sum(IoUs)/len(IoUs))
    # print('Rec Center Accuracy:')
    # print(sum(center_accuracy)/len(center_accuracy)*100)
    print('REC Acc@0.5')
    print(count_above_margin(IoUs)/len(IoUs) * 100)
    print('Consistency:')
    print(CR/(CR+CM+WR)*100)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--qwen", type=bool, default=False)
    parser.add_argument("--internvl", type=bool, default=False)
    parser.add_argument("--v7w", type=bool, default=False)
    args = parser.parse_args()

    eval_model(args)
