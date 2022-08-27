# Text to Image を行うスクリプト
# promptを編集して利用
# 参考ドキュメント: https://huggingface.co/blog/stable_diffusion
import torch
from diffusers import StableDiffusionPipeline
from torch import autocast
import time
import os
from dotenv import load_dotenv
load_dotenv()

MODEL_ID = "CompVis/stable-diffusion-v1-4"
DEVICE = "cuda"
YOUR_TOKEN = os.getenv('YOUR_TOKEN')

pipe = StableDiffusionPipeline.from_pretrained(
    MODEL_ID, revision="fp16", torch_dtype=torch.float16, use_auth_token=YOUR_TOKEN)
pipe.to(DEVICE)

prompt = "new League of Legends champion, use japanese blade, rich detail face and eyes, detailed CG art"

with autocast(DEVICE):
  # generator = torch.Generator("cuda").manual_seed(2525) # generator が必要な際には利用 generator=generator で引数に入れる
  image = pipe(prompt, guidance_scale=7.5,
               num_inference_steps=100)["sample"][0]

  os.makedirs('./results', exist_ok=True)
  ut = int(time.time())
  image.save(f"./results/{ut}.png")

  f = open(f'./results/{ut}.txt', 'w')
  f.write(f'{prompt}')
  f.close()
