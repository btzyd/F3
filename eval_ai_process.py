
import json
import os
from llava.eval.m4c_evaluator import EvalAIAnswerProcessor
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--json_dir', required=True, type=str) 
args = parser.parse_args()

answer_processor = EvalAIAnswerProcessor()

input_dir = args.json_dir

src = json.load(open(os.path.join(input_dir, "clean_result_original.json"), "r"))

res = []
for item in src:
    res.append({
        "question_id": item["question_id"], 
        "answer": answer_processor(item["answer"])
    })

json.dump(res, open(os.path.join(input_dir, "clean_result.json"), "w"))

src = json.load(open(os.path.join(input_dir, "adv_result_original.json"), "r"))

res = []
for item in src:
    res.append({
        "question_id": item["question_id"], 
        "answer": answer_processor(item["answer"])
    })

json.dump(res, open(os.path.join(input_dir, "adv_result.json"), "w"))

src = json.load(open(os.path.join(input_dir, "purify_result_original.json"), "r"))

res = []
for item in src:
    res.append({
        "question_id": item["question_id"], 
        "answer": answer_processor(item["answer"])
    })

json.dump(res, open(os.path.join(input_dir, "purify_result.json"), "w"))
