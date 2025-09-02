import torch
from torchvision import transforms
from llava.mm_utils import get_model_name_from_path


class LLaVA_F3():
    def __init__(self, model_path, tokenizer, model, image_processor, device, f3_alpha, f3_beta, f3_v3):
        model_name = get_model_name_from_path(model_path)
        self.tokenizer = tokenizer
        self.model = model
        self.image_processor = image_processor
        self.device = device
        conv_mode = "vicuna_v1"

        self.input_ids = None
        self.image_sizes = None

        OPENAI_CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
        OPENAI_CLIP_STD = [0.26862954, 0.26130258, 0.27577711]
        
        self.f3_alpha = f3_alpha
        self.f3_beta = f3_beta
        self.f3_v3 = f3_v3

        self.normalizer = transforms.Compose([transforms.Normalize(mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD),])

        if 'plain' in model_name and 'finetune' not in model_name.lower() and 'mmtag' not in conv_mode:
            conv_mode = conv_mode + '_mmtag'
        print(f'It seems that this is a plain model, but it is not using a mmtag prompt, auto switching to {conv_mode}.')

    
    def eval(self):
        self.model.eval()

    def requires_grad_(self, bool_var):
        self.model.requires_grad_(bool_var)
    
    def predict(self, image_tensor, input_ids, image_sizes):
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=self.normalizer(image_tensor).to(dtype=torch.float16, device=self.device, non_blocking=True),
                image_sizes=image_sizes,
                do_sample=False,
                temperature=0,
                top_p=None,
                num_beams=1,
                max_new_tokens=128,
                use_cache=True)
        outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        return outputs
    
    def get_logit(self, image_tensor, ):
        outputs = self.model(
            self.input_ids,
            images=self.normalizer(image_tensor).to(dtype=torch.float16, device=self.device, non_blocking=True),
            image_sizes=self.image_sizes,
            use_cache=False,
            return_dict=True)
        return outputs.logits[:, -1, :]

    def get_attention(self, image_tensor, input_ids, image_sizes=None):
        if image_sizes is None:
            image_sizes = self.image_sizes
        outputs = self.model.generate(
                input_ids,
                images=self.normalizer(image_tensor).to(dtype=torch.float16, device=self.device, non_blocking=True),
                image_sizes=image_sizes,
                do_sample=False,
                temperature=0,
                top_p=None,
                num_beams=1,
                max_new_tokens=128,
                use_cache=True,
                output_attentions=True,
                return_dict_in_generate=True)

        attention = torch.cat(outputs.attentions[0], dim=0)[:,:,-1,35:35+576]
        return attention
    
    def purify_f3(self, image):
        pgd_eps = self.f3_beta
        image_adv = image.detach().clone().to(image.device)
        self.model.zero_grad()
        ref_image = image_adv.detach().clone().to(image_adv.device) + (torch.randint(-int(self.f3_alpha*255), int(self.f3_alpha*255)+1, image_adv.shape)/255).to(image_adv.device)
        ref_image = torch.clamp(ref_image, min=0, max=1)
        ref_attention = self.get_attention(ref_image, self.input_ids, self.image_sizes)
        image_adv.requires_grad = True
        adv_attention = self.get_attention(image_adv, self.input_ids, self.image_sizes)
        loss_before_mean = (adv_attention.contiguous()-ref_attention.contiguous()) ** 2
        loss = torch.mean(loss_before_mean)*10000
        grad = torch.autograd.grad(loss, image_adv)[0]
        grad_sign = grad.sign()
        if self.f3_v3:
            grad_max = torch.max(grad)
            grad_min = torch.min(grad)
            grad_normalized = (grad - grad_min) / (grad_max - grad_min)
            grad_normalized = torch.clamp(grad_normalized/torch.mean(grad_normalized), min=0, max=1)
            image_adv = image_adv - torch.round(grad_normalized*self.f3_beta*255) * grad_sign / 255
        else:
            image_adv = image_adv - grad_sign * self.f3_beta
            
        image_adv = image + torch.clamp(image_adv - image, min=-pgd_eps, max=pgd_eps)
        image_adv = torch.clamp(image_adv, min=0, max=1)

        return image_adv

    
    def get_logit_with_purify_f3(self, x):
        x_purify = self.purify_f3(x)
        logits = self.get_logit(x_purify)
        return logits
   