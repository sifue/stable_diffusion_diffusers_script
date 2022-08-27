# Text to Image を複数回行いグリッド化するスクリプト
# promptを編集して利用
# カラム数は増やしてもよいがサイズが512x512に収まるようにする必要がある
# 参考ドキュメント: https://huggingface.co/blog/stable_diffusion
from PIL import Image
import torch
from diffusers import StableDiffusionPipeline
from torch import autocast
import time
import os
from dotenv import load_dotenv
load_dotenv()


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols*w, i//cols*h))
    return grid


MODEL_ID = "CompVis/stable-diffusion-v1-4"
DEVICE = "cuda"
YOUR_TOKEN = os.getenv('YOUR_TOKEN')

pipe = StableDiffusionPipeline.from_pretrained(
    MODEL_ID, revision="fp16", torch_dtype=torch.float16, use_auth_token=YOUR_TOKEN)
pipe.to(DEVICE)

num_images = 4
prompt = ["beautiful illustration of anime maid, stunning and rich detail, pretty face and eyes. 3D style, Pixiv featured."] * num_images

with autocast(DEVICE):
  generator = torch.Generator("cuda").manual_seed(25254)
  images = pipe(prompt,
                guidance_scale=7.5,
                generator=generator,
                num_inference_steps=250, height=256, width=256)["sample"]
  grid = image_grid(images, rows=2, cols=2)

  os.makedirs('./results', exist_ok=True)
  ut = int(time.time())
  grid.save(f"./results/{ut}.png")

  f = open(f'./results/{ut}.txt', 'w')
  f.write(f'{prompt}')
  f.close()
