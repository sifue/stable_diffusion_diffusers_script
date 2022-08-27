# WSL2で Diffusers を利用した Stable Diffusion の実行スクリプト
- Text to Imageのスクリプト (`t2i.py`)
- Image to Imageのスクリプト (`i2i.py`)
以上のスクリプトを用意。

## 環境構築と実行方法
WindowsのWSL2を利用することを想定。RTX2070を利用して実行テスト済み。

1. [WSL2 上の Ubuntu での NVIDIA CUDA ツールキット, NVIDIA cuDNN, PyTorch, TensorFlow 2.7 のインストールと動作確認（Windows 上）](https://www.kkaneko.jp/tools/wsl/wsl_tensorflow2.html)
1. [【簡単】ローカル環境でStable Diffusionを実行する方法](https://self-development.info/%e3%80%90%e7%b0%a1%e5%8d%98%e3%80%91%e3%83%ad%e3%83%bc%e3%82%ab%e3%83%ab%e7%92%b0%e5%a2%83%e3%81%a7stable-diffusion%e3%81%a7%e5%ae%9f%e8%a1%8c%e3%81%99%e3%82%8b%e6%96%b9%e6%b3%95/)

以上を参考にインストール。

`.env` ファイルで

```
YOUR_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

のようにHugging Faceのトークンを設定することを想定。あとは

```
python3 t2i.py
```

などと実行するだけでOK。必要に応じてスクリプトを編集。

## 参考ドキュメント

- [Stable Diffusion with 🧨 Diffusers](https://huggingface.co/blog/stable_diffusion)
- [image-2-image using diffusers](https://colab.research.google.com/github/patil-suraj/Notebooks/blob/master/image_2_image_using_diffusers.ipynb#scrollTo=V24njWQBC8eC)
