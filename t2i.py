# Text to Image を行うスクリプト
# --seed 引数でシード値を渡せる
# promptを編集して利用
# 参考ドキュメント: https://huggingface.co/blog/stable_diffusion
import torch
from diffusers import StableDiffusionPipeline
from torch import autocast
import time
import os
from dotenv import load_dotenv
load_dotenv()

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--seed", help="optional")
args = parser.parse_args()
seed = int(args.seed) if args.seed else 2525

MODEL_ID = "CompVis/stable-diffusion-v1-4"
DEVICE = "cuda"
YOUR_TOKEN = os.getenv('YOUR_TOKEN')

pipe = StableDiffusionPipeline.from_pretrained(
    MODEL_ID, revision="fp16", torch_dtype=torch.float16, use_auth_token=YOUR_TOKEN)
pipe.to(DEVICE)

prompt = "A photo of a young, slender woman wearing shiny rubber tights standing facing forward, showing her entire body."
# prompt = "Medium shot, alone, hq, sharp eyes, pictures of beautiful people cosplaying at Comiket, detail, pretty face and eyes, best shot"
# prompt = "Hatsune Mikum, Medium shot, alone, an anime girl, Otaku, Daisuki, Senpai, Kawaii, hq, wallpaper, style of Moe, VTuber, Manga, character introduction, sharp eyes,best scene, import, official, capture, winning works, winning creative, best illustration, ranking, support artist, angle, how to draw, demo, import, comic con, expo, gallery, art book"
# prompt = "Young, slender, well-waisted woman in tights, facing front."
# prompt = "beautiful illustration of anime maid, stunning and rich detail, pretty face and eyes. 3D style, Pixiv featured."

with autocast(DEVICE):

  print(f'Generating start. seed: {seed}')
  generator = torch.Generator("cuda").manual_seed(seed)
  image = pipe(prompt, guidance_scale=7.5,
               generator=generator,
               num_inference_steps=250)["sample"][0]

  os.makedirs('./results', exist_ok=True)
  ut = int(time.time())
  image.save(f"./results/{ut}.png")

  f = open(f'./results/{ut}.txt', 'w')
  f.write(f'{seed}\n')
  f.write(f'{prompt}\n')
  f.close()
  print(f'Generating finished. seed: {seed}')
