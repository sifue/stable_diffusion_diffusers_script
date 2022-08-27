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

prompt = "beautiful illustration of anime maid, stunning and rich detail, pretty face and eyes. 3D style, Pixiv featured."

with autocast(DEVICE):
  # generator が必要な際には利用 generator=generator で引数に入れる
  generator = torch.Generator("cuda").manual_seed(25253)
  image = pipe(prompt, guidance_scale=7.5,
               generator=generator,
               num_inference_steps=250)["sample"][0]

  os.makedirs('./results', exist_ok=True)
  ut = int(time.time())
  image.save(f"./results/{ut}.png")

  f = open(f'./results/{ut}.txt', 'w')
  f.write(f'{prompt}')
  f.close()
