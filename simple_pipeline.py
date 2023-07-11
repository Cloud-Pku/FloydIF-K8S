from deepfloyd_if.modules import IFStageI, IFStageII, StableStageIII
from deepfloyd_if.modules.t5 import T5Embedder

device = 'cuda:0'
if_I = IFStageI('IF-I-XL-v1.0', device=device)
if_II = IFStageII('IF-II-L-v1.0', device=device)
if_III = StableStageIII('stable-diffusion-x4-upscaler', device=device)
t5 = T5Embedder(device="cpu")

from deepfloyd_if.pipelines import dream

prompt = """a orange and white medium format 85mm portrait of a kitten wearing a tuxedo on his way to a funeral, the image is high quality and highly detailed with the kitten's features clearly visible, photographer Edward Weston used Agfa Isopan ISO 25 film to create this image, which resembles Edward Weston's photograph Pepper No. 35"""
neg_prompt = None #'logo, text, watermark, word, signature, label, sign, meme'
count = 9

result = dream(
    t5=t5, if_I=if_I, if_II=if_II, if_III=if_III,
    prompt=[prompt]*count,
    seed=42,
    negative_prompt = neg_prompt,
    if_I_kwargs={
        "guidance_scale": 7.0,
        "sample_timestep_respacing": "smart100",
    },
    if_II_kwargs={
        "guidance_scale": 4.0,
        "sample_timestep_respacing": "smart50",
    },
    if_III_kwargs={
        "guidance_scale": 9.0,
        "noise_level": 20,
        "sample_timestep_respacing": "75",
    },
)
if_I.show(result['I'], size=14,fname='figI')
if_II.show(result['II'], size=14,fname='figII')
if_III.show(result['III'], size=14,fname='figIII')