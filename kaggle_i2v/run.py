"""KAGGLE GPU İŞİ (tolun-farm i2v): job.json'daki her sabit görseli GERÇEK hareketli klibe çevirir (görüntü→video).
Model: Lightricks LTX-Video (diffusers LTXImageToVideoPipeline). Çıktı: /kaggle/working/out/<id>.mp4 (+ log.json).
Görseller tolun-farm artgen-cache dalından (herkese açık raw URL) indirilir. Kişisel bilgi/anahtar YOK."""
import json, os, sys, time, subprocess, urllib.request, io
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "diffusers>=0.32", "transformers", "accelerate", "sentencepiece", "imageio[ffmpeg]"], check=False)
import torch
from PIL import Image
from diffusers import LTXImageToVideoPipeline
from diffusers.utils import export_to_video
J = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "job.json")))
OUT = "/kaggle/working/out"; os.makedirs(OUT, exist_ok=True); log = []
RAW = "https://raw.githubusercontent.com/darkpegasuslord28-cpu/tolun-farm/artgen-cache/artgen/"
pipe = LTXImageToVideoPipeline.from_pretrained(J.get("model", "Lightricks/LTX-Video"), torch_dtype=torch.float16)
pipe.enable_model_cpu_offload(); pipe.vae.enable_tiling()
NEG = "worst quality, inconsistent motion, blurry, jittery, distorted, text, watermark, people, morphing"
for it in J["items"]:
    t0 = time.time()
    try:
        img = Image.open(io.BytesIO(urllib.request.urlopen(RAW + it["image"], timeout=120).read())).convert("RGB")
        w, h = it.get("size", J.get("size", [512, 768]))
        img = img.resize((w, h), Image.LANCZOS)
        g = torch.Generator("cuda").manual_seed(it.get("seed", 7))
        fr = pipe(image=img, prompt=it["prompt"], negative_prompt=NEG, width=w, height=h, num_frames=it.get("frames", J.get("frames", 97)),
                  num_inference_steps=it.get("steps", J.get("steps", 40)), guidance_scale=3.0, generator=g).frames[0]
        export_to_video(fr, f"{OUT}/{it['id']}.mp4", fps=24); log.append({"id": it["id"], "ok": True, "sec": round(time.time() - t0)})
    except Exception as e:
        log.append({"id": it["id"], "ok": False, "err": repr(e)[:300]})
    json.dump(log, open(f"{OUT}/log.json", "w"), indent=1); print(log[-1], flush=True)
