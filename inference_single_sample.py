# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# Copyright 2024 Black Forest Labs and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tempfile
from PIL import Image
import subprocess

import torch
# import gradio as gr
import string
import random, time, os, math   

from src.flux.generate import generate_from_test_sample, seed_everything
from src.flux.pipeline_tools import CustomFluxPipeline, load_modulation_adapter, load_dit_lora
from src.utils.data_utils import get_train_config, image_grid, pil2tensor, json_dump, pad_to_square, cv2pil, merge_bboxes
from eval.tools.face_id import FaceID
from eval.tools.florence_sam import ObjectDetector
import shutil
import yaml
import numpy as np

import argparse

dtype = torch.bfloat16
device = "cuda"

config_path = "train/config/XVerse_config_demo.yaml"

config = config_train = get_train_config(config_path)
config["model"]["dit_quant"] = "int8-quanto"
config["model"]["use_dit_lora"] = False
model = CustomFluxPipeline(
    config, device, torch_dtype=dtype,
)
model.pipe.set_progress_bar_config(leave=False)

# face_model = FaceID(device)
# detector = ObjectDetector(device)

config = get_train_config(config_path)
model.config = config

run_mode = "mod_only" # orig_only, mod_only, both
store_attn_map = False
run_name = time.strftime("%m%d-%H%M")

num_inputs = 6

ckpt_root = "./checkpoints/XVerse"
model.clear_modulation_adapters()
model.pipe.unload_lora_weights()
if not os.path.exists(ckpt_root):
    print("Checkpoint root does not exist.")

modulation_adapter = load_modulation_adapter(model, config, dtype, device, f"{ckpt_root}/modulation_adapter", is_training=False)
model.add_modulation_adapter(modulation_adapter)
if config["model"]["use_dit_lora"]:
    load_dit_lora(model, model.pipe, config, dtype, device, f"{ckpt_root}", is_training=False)


def generate_image(prompt, cond_size, target_height, target_width, seed, vae_skip_iter, control_weight_lambda,      latent_dblora_scale_str, latent_sblora_scale_str, vae_lora_scale_str,
                   indexs, num_images, *args):  # 新增 num_images 参数
    torch.cuda.empty_cache()
    # 使用传入的 num_images
    # num_images = 4

    # 从 args 中提取 images, captions, idips_checkboxes
    images = list(args[:len(indexs)])
    captions = list(args[len(indexs):2*len(indexs)])
    idips_checkboxes = list(args[2*len(indexs):3*len(indexs)])

    print(f"Length of images: {len(images)}")
    print(f"Length of captions: {len(captions)}")
    print(f"Indexs: {indexs}")
    
    print(f"Control weight lambda: {control_weight_lambda}")
    if control_weight_lambda != "no":
        parts = control_weight_lambda.split(',')
        new_parts = []
        for part in parts:
            if ':' in part:
                left, right = part.split(':')
                values = right.split('/')
                # 保存整体值
                global_value = values[0]
                id_value = values[1]
                ip_value = values[2]
                new_values = [global_value]
                for is_id in idips_checkboxes:
                    if is_id:
                        new_values.append(id_value)
                    else:
                        new_values.append(ip_value)
                new_part = f"{left}:{('/'.join(new_values))}"
                new_parts.append(new_part)
            else:
                new_parts.append(part)
        control_weight_lambda = ','.join(new_parts)
    
    print(f"Control weight lambda: {control_weight_lambda}")

    src_inputs = []
    use_words = []
    for i, (image_path, caption) in enumerate(zip(images, captions)):
        if image_path:
            if caption.startswith("a ") or caption.startswith("A "):
                word = caption[2:]
            else:
                word = caption
            
            if f"ENT{i+1}" in prompt:
                prompt = prompt.replace(f"ENT{i+1}", caption)
            
            # 移除图片调整大小和保存操作
            input_image_path = image_path

            src_inputs.append(
                {
                    "image_path": input_image_path,
                    "caption": caption
                }
            )
            use_words.append((i, word, word))


    test_sample = dict(
        input_images=[], position_delta=[0, -32], 
        prompt=prompt,
        target_height=target_height,
        target_width=target_width,
        seed=seed,
        cond_size=cond_size,
        vae_skip_iter=vae_skip_iter,
        lora_scale=latent_dblora_scale_str,
        control_weight_lambda=control_weight_lambda,
        latent_sblora_scale=latent_sblora_scale_str,
        condition_sblora_scale=vae_lora_scale_str,
        double_attention=False,
        single_attention=True,
    )
    if len(src_inputs) > 0:
        test_sample["modulation"] = [
            dict(
                type="adapter",
                src_inputs=src_inputs,
                use_words=use_words,
            ),
        ]
    
    target_size = int(round((target_width * target_height) ** 0.5) // 16 * 16)
    model.config["train"]["dataset"]["val_condition_size"] = cond_size
    model.config["train"]["dataset"]["val_target_size"] = target_size
    
    if control_weight_lambda == "no":
        control_weight_lambda = None
    if vae_skip_iter == "no":
        vae_skip_iter = None
    use_condition_sblora_control = True
    use_latent_sblora_control = True
    image = generate_from_test_sample(
        test_sample, model.pipe, model.config, 
        num_images=num_images,  # 使用传入的 num_images
        target_height=target_height,
        target_width=target_width,
        seed=seed,
        store_attn_map=store_attn_map, 
        vae_skip_iter=vae_skip_iter,  # 使用新的参数
        control_weight_lambda=control_weight_lambda,  # 传递新的参数
        double_attention=False,  # 新增参数
        single_attention=True,  # 新增参数
        latent_dblora_scale=latent_dblora_scale_str,
        use_latent_sblora_control=use_latent_sblora_control,
        latent_sblora_scale=latent_sblora_scale_str,
        use_condition_sblora_control=use_condition_sblora_control,
        condition_sblora_scale=vae_lora_scale_str,
    )
    if isinstance(image, list):
        num_cols = 2
        num_rows = int(math.ceil(num_images / num_cols))  # 使用传入的 num_images
        image = image_grid(image, num_rows, num_cols)

    return image

def main():
    parser = argparse.ArgumentParser(description='XVerse Inference')
    parser.add_argument('--prompt', type=str, required=True, help='Prompt for image generation')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--cond_size', type=int, default=256, help='Condition size')
    parser.add_argument('--target_height', type=int, default=768, help='Generated image height')
    parser.add_argument('--target_width', type=int, default=768, help='Generated image width')
    parser.add_argument('--weight_id', type=float, default=3, help='Weight for ID')
    parser.add_argument('--weight_ip', type=float, default=5, help='Weight for IP')
    parser.add_argument('--latent_lora_scale', type=float, default=0.85, help='Latent lora scale')
    parser.add_argument('--vae_lora_scale', type=float, default=1.3, help='VAE lora scale')
    parser.add_argument('--vae_skip_iter_s1', type=float, default=0.05, help='VAE skip iter before')
    parser.add_argument('--vae_skip_iter_s2', type=float, default=0.8, help='VAE skip iter after')
    parser.add_argument('--images', nargs='+', help='List of image paths')
    parser.add_argument('--captions', nargs='+', help='List of captions corresponding to images')
    parser.add_argument('--idips', nargs='+', type=lambda x: (str(x).lower() == 'true'), help='List of ID/IP flags')
    parser.add_argument('--save_path', type=str, default="generated_image.png", help='Path to save the generated image')
    parser.add_argument('--num_images', type=int, default=4, help='Number of images to generate')

    args = parser.parse_args()

    # 验证输入参数
    if args.images and args.captions and len(args.images) != len(args.captions):
        raise ValueError("Number of images and captions must be the same")
    if args.images and args.idips and len(args.images) != len(args.idips):
        raise ValueError("Number of images and ID/IP flags must be the same")

    # 计算 control_weight_lambda 和 vae_skip_iter
    control_weight_lambda = f"0-1:1/{args.weight_id}/{args.weight_ip}"
    vae_skip_iter = f"0-{args.vae_skip_iter_s1}:1,{args.vae_skip_iter_s2}-1:1"
    latent_sblora_scale_str = f"0-1:{args.latent_lora_scale}"
    latent_dblora_scale_str = f"0-1:{args.latent_lora_scale}"
    vae_lora_scale_str = f"0-1:{args.vae_lora_scale}"

    # 准备 indexs
    indexs = list(range(len(args.images))) if args.images else []

    image = generate_image(
        args.prompt,
        args.cond_size,
        args.target_height,
        args.target_width,
        args.seed,
        vae_skip_iter,
        control_weight_lambda,
        args.latent_lora_scale,
        latent_sblora_scale_str,
        vae_lora_scale_str,
        indexs,
        args.num_images,  # 传递 num_images 参数
        *args.images,
        *args.captions,
        *args.idips
    )

    # 使用命令行传入的路径保存生成的图像
    image.save(args.save_path)
    print(f"Generated image saved to {args.save_path}")

if __name__ == "__main__":
    main()
