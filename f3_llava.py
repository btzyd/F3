import argparse
import os
import numpy as np
import random
import time
import datetime
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torchvision
from datasets import vqa_dataset_llava, save_result
from models import LLaVA_F3
import utils
from attacks import AutoAttack_LVLM

from llava.mm_utils import get_model_name_from_path
from llava.model.builder import load_pretrained_model

def main(args):
    utils.init_distributed_mode(args)    
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    model_path = os.path.join(args.huggingface_root, "llava-v1.5-7b")
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, None, model_name, device_map=device, device=device)
    image_processor.do_normalize = False

    #### Dataset #### 
    print("Creating vqa datasets")

    datasets = vqa_dataset_llava(args.annotation_file, args.dataset_root_dir, tokenizer, image_processor, model.config, conv_mode="vicuna_v1")

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()               
        sampler = torch.utils.data.DistributedSampler(datasets, num_replicas=num_tasks, rank=global_rank, shuffle=False)
    else:
        sampler = None

    data_loader = DataLoader(datasets, batch_size=1, num_workers=4, pin_memory=True, sampler=sampler, shuffle=False, collate_fn=None, drop_last=False)              

    print("Creating model")
    
    model = LLaVA_F3(model_path, tokenizer, model, image_processor, device, args.f3_alpha, args.f3_beta, args.f3_v3)
    
    if args.attack_method=="unadaptive":
        attack = AutoAttack_LVLM(model.get_logit, eps=args.aa_eps, device=device, version="standard")
    elif args.attack_method=="adaptive":
        attack = AutoAttack_LVLM(model.get_logit_with_purify_f3, eps=args.aa_eps, device=device, version="rand")
    else:
        attack = None

    print("Start training")
    start_time = time.time()    


    metric_logger = utils.MetricLogger(delimiter="  ")
    header = f"Attacking by {args.attack_method}:"
    print_freq = 5
    
    clean_result = []
    attack_result = []
    purify_result = []


    for n, (image_path, image, question, question_id, image_size, ori_question) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):        
        image = image.to(device=device, non_blocking=True)
        clean_image = image.to(dtype=torch.float16, device=device, non_blocking=True)
        question = question.to(device=device)
        image_size = image_size
        clean_answer = model.predict(clean_image, question, image_size)
        clean_result.append({
            "question_id": int(question_id.numpy()[0]), 
            "answer": clean_answer
            })
        
        torchvision.utils.save_image(image, os.path.join(args.image_dir, "{}_clean.png".format(image_path[0])))

        if attack:
            model.input_ids = question.to(device=device).long()
            model.input_ids.requires_grad = False
            model.image_sizes = image_size
            image.requires_grad = True
   
            clean_logit = model.get_logit(image)
            
            clean_label = torch.argmax(clean_logit, dim=1).to(device)
            adv_image = attack.run_standard_evaluation(image, clean_label, bs=1, return_labels=False, state_path=None, strong=True)
                
    
            attack_answer = model.predict(adv_image, question, image_size)
            purify_image = model.purify_f3(adv_image)
            purify_answer = model.predict(purify_image, question, image_size)
            attack_result.append({
                "question_id": int(question_id.numpy()[0]), 
                "answer": attack_answer
                    })
            torchvision.utils.save_image(adv_image, os.path.join(args.image_dir, "{}_adv.png".format(image_path[0])))
            purify_result.append({
                "question_id": int(question_id.numpy()[0]), 
                "answer": purify_answer
            })

            torchvision.utils.save_image(purify_image, os.path.join(args.image_dir, "{}_purify.png".format(image_path[0])))
    
    save_result(clean_result, args.output_dir, 'clean_result_original')
    if len(attack_result)>0:
        save_result(attack_result, args.output_dir, 'adv_result_original')
        save_result(purify_result, args.output_dir, 'purify_result_original')

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    parser = argparse.ArgumentParser()
    
    # default config
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')    
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)
    
    # model config
    parser.add_argument('--huggingface_root', default="/root/huggingface_model", type=str)
   
    # dataset config
    parser.add_argument('--annotation_file', required=True, type=str)
    parser.add_argument('--dataset_root_dir', default="/root/nfs/dataset/val2014", type=str)

    # attack config
    parser.add_argument('--attack_method', type=str, required=True, choices=["none", "unadaptive", "adaptive"])
    
    # f3 confg
    parser.add_argument('--f3_alpha', default=16, type=int)
    parser.add_argument('--f3_beta', default=32, type=int)
    parser.add_argument('--f3_v3', action="store_true")

    # aa config
    parser.add_argument('--aa_eps', type=int, default=16)

    # output config
    parser.add_argument('--output_dir', required=True, type=str) 

    args = parser.parse_args()

    args.f3_alpha = float(args.f3_alpha)/255
    args.f3_beta = float(args.f3_beta)/255
    args.aa_eps = float(args.aa_eps)/255
    
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    args.image_dir = os.path.join(args.output_dir, "image")
    Path(args.image_dir).mkdir(parents=True, exist_ok=True)

    main(args)
