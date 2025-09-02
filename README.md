## Fighting Fire with Fire (F3): A Training-free and Efficient Visual Adversarial Example Purification Method in LVLMs

This is a PyTorch implementation of the [F3 paper](https://arxiv.org/abs/2506.01064), which is accepted by ACMMM 2025 as Oral.

## Preparing the environment, code, data and model

1. Prepare the environment.

Creating a python environment and activate it via the following command.

```bash
conda create -n f3 python==3.10
conda activate f3
pip install --upgrade pip
pip install -e .
pip install git+https://github.com/fra31/auto-attack
pip install protobuf
```

2. Clone this repository.

```bash
git clone https://github.com/btzyd/F3.git
```

3. Prepare the dataset VQA v2.

We have provided 1000 VQA v2 questions in [vqav2_1000.json](annotation/vqav2_1000.json). As for images, you can download [COCO-val2014](http://images.cocodataset.org/zips/val2014.zip). 

4. Download models.

You can download [LLaVA-v1.5-7B](https://huggingface.co/liuhaotian/llava-v1.5-7b). Of course, you can also download the model while loading it in python code, though the download may be unstable. We recommend that you download the model first and then run the python code to load it from local directories.

5. Modify Codes.

There are one modification to the source code:

  - Comment out ["@torch.no_grad()"](https://github.com/huggingface/transformers/blob/345b9b1a6a308a1fa6559251eb33ead2211240ac/src/transformers/generation/utils.py#L1173) in the installed transformer library to provide gradient for attention.

## Run the F3 code

### Attack and Purify


For unadaptive autoattack:

```bash
python -m torch.distributed.run \
    --nproc_per_node 8 --nnodes 1 --node_rank 0 \
    --master_addr 127.0.0.1 --master_port 25031 \
    f3_llava.py --annotation_file vqav2_1000.json --attack_method unadaptive --aa_eps 16 --f3_v3 \
    --output_dir output_llava_7b/unadaptive_eps16
```

For adaptive autoattack:
```bash
python -m torch.distributed.run \
    --nproc_per_node 8 --nnodes 1 --node_rank 0 \
    --master_addr 127.0.0.1 --master_port 25031 \
    f3_llava.py --annotation_file vqav2_1000.json --attack_method adaptive --aa_eps 8 --f3_v3 \
    --output_dir output_llava_7b/adaptive_eps8
```

The meaning of the parameters are as follows:

- `huggingface_root`: The path of LLaVA-v1.5-7B.
- `dataset_root_dir`: The path of COCO 2014 images.
- `annotation_file`: The path of annotation file.
- `attack_method`: Use adaptive or non-adaptive attacks.
- `aa_eps`: The $\epsilon_\infty$ of autoattack.
- `f3_v3`: Use F3-v2 or F3-v3.

### Evaluate

First, preprocess the output file.

```bash
python eval_ai_process.py --json_dir output_llava_7b/adaptive_eps8/
```

The meaning of the parameters are as follows:

- ``json_dir``: The path of output file.

Second, run the evaluation code. Note that the evaluation code must be executed in a Python2 environment.

```bash
python eval_f3.py output_llava_7b/adaptive_eps8/
```

### Result

F3-v3 result on LLaVA-v1.5-7B:

Attack method|Clean|Adversarial|Purify
:--:|:--:|:--:|:--:|
Unadaptive ($\epsilon_\infty=16$)|76.44|16.12|56.61
Adaptive ($\epsilon_\infty=8$)|76.44|-|62.70

## Citation
Since the conference proceedings have not yet been published, this is the citation format for the arXiv version.

```bibtex
@article{zhang2025fighting,
  title={Fighting Fire with Fire (F3): A Training-free and Efficient Visual Adversarial Example Purification Method in LVLMs},
  author={Zhang, Yudong and Xie, Ruobing and Huang, Yiqing and Chen, Jiansheng and Sun, Xingwu and Kang, Zhanhui and Wang, Di and Wang, Yu},
  journal={arXiv preprint arXiv:2506.01064},
  year={2025}
}
```