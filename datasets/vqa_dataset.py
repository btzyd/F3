import os
import json
from PIL import Image
from torch.utils.data import Dataset
from .data_utils import pre_question
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates
from llava.mm_utils import tokenizer_image_token, process_images


class vqa_dataset_llava(Dataset):
    def __init__(self, annotation_file, image_dir, tokenizer, image_processor, model_config, conv_mode):
        self.annotation = json.load(open(os.path.join("annotation", annotation_file), 'r'))
        self.image_dir = image_dir

        self.conv_mode = conv_mode
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config

    def __len__(self):
        return len(self.annotation)
    
    def __getitem__(self, index):    
        image_path_name = self.annotation[index]["image_name"]
        ori_question = pre_question(self.annotation[index]["question"])
        question_id = self.annotation[index]["question_id"]
        image_path = os.path.join(self.image_dir, image_path_name)
        image = Image.open(image_path).convert('RGB')
               
        if self.model_config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + ori_question
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + ori_question

        qs = qs + "\nAnswer the question using a single word or phrase."

        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        image_tensor = process_images([image], self.image_processor, self.model_config)[0]
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')

        return image_path_name.split(".")[0], image_tensor, input_ids, question_id, image.size, ori_question
