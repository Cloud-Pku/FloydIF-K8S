
from deepfloyd_if.modules.t5 import T5Embedder
from datetime import datetime

import torch
from deepfloyd_if.floyd import FloydDM
import tqdm



device = 'cpu'#'cuda:0'
if_I = FloydDM('IF-I-M-v1.0', device=device, cache_dir='/mnt/nfs/share/deepfloydIF/weights/IF_')
t5 = T5Embedder(device="cpu", cache_dir='/mnt/nfs/share/deepfloydIF/weights/IF_')
prompt = '''a white medium format 85mm portrait of a kitten, the image is high quality and highly detailed with the kitten's features clearly visible'''
# prompt = """a black and white medium format 85mm portrait of a kitten wearing a tuxedo on his way to a funeral, the image is high quality and highly detailed with the kitten's features clearly visible, photographer Edward Weston used Agfa Isopan ISO 25 film to create this image, which resembles Edward Weston's photograph Pepper No. 35"""
neg_prompt = None #'logo, text, watermark, word, signature, label, sign, meme'
count = 9
aspect_ratio='1:1'
disable_watermark=True
prompt=[prompt]*count
seed = 1
progress=True
if seed is None:
        seed = int((datetime.utcnow().timestamp() * 10 ** 6) % (2 ** 32 - 1))
if_I.seed_everything(seed)

t5_embs = t5.get_text_embeddings(prompt)


# construct args
if_I_kwargs={
    "guidance_scale": 7.0,
    "sample_timestep_respacing": "smart100",
}
if_I_kwargs['seed'] = seed
if_I_kwargs['t5_embs'] = t5_embs
if_I_kwargs['aspect_ratio'] = aspect_ratio
if_I_kwargs['progress'] = progress

stageI_generations, _, mid_img, x0_pred, step_num, mean_img, pred_eps = if_I.embeddings_to_image(**if_I_kwargs)
pil_images_I = if_I.to_images(stageI_generations[:9], disable_watermark=disable_watermark)

if_I.savefig(pil_images_I, fname='final')
print("start drawing...")
for i in step_num:
    print(f'drawing {i}th...')
    mid_i = if_I.to_images(mid_img[i][:9], disable_watermark=disable_watermark)
    x0_i = if_I.to_images(x0_pred[i][:9], disable_watermark=disable_watermark)
    mean_i = if_I.to_images(mean_img[i][:9], disable_watermark=disable_watermark)
    pred_eps_i = if_I.to_images(pred_eps[i][:9], disable_watermark=disable_watermark)
    if_I.savefig(mid_i, fname=f'mid_{i}')
    if_I.savefig(x0_i, fname=f'x0_pred_{i}')
    if_I.savefig(mean_i, fname=f'mean_{i}')
    if_I.savefig(pred_eps_i, fname=f'pred_eps_{i}')