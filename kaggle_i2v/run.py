"""KAGGLE GPU İŞİ (tolun-farm i2v): job.json'daki her sabit görseli GERÇEK hareketli klibe çevirir (görüntü→video).
Modeller: "ltx" = Lightricks LTX-Video (diffusers LTXImageToVideoPipeline) · "wan" = Wan2.2-TI2V-5B (Apache-2.0, diffusers).
Çıktı: /kaggle/working/out/<id>.mp4 (+ log.json, hareket ölçüsü). Görseller tolun-farm artgen-cache'ten. Kişisel bilgi/anahtar YOK."""
import json, os, sys, time, subprocess, urllib.request, io
J = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "job.json")))
if J.get("mode") == "blender":   # 3B ÜRÜN SAHNELERİ (Blender/bpy, Cycles GPU): kaggle_i2v/tolun3d.py — her sahnede aynı ürün
    OUT = "/kaggle/working/out"; os.makedirs(OUT, exist_ok=True); log = []
    subprocess.run("apt-get install -y -qq libxi6 libxxf86vm1 libxfixes3 libxrender1 libgl1 libxkbcommon0 libsm6 > /dev/null 2>&1", shell=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "bpy", "imageio-ffmpeg"], check=False)
    import imageio_ffmpeg; FF = imageio_ffmpeg.get_ffmpeg_exe()
    sc = "/kaggle/working/" + J.get("script", "tolun3d.py")
    urllib.request.urlretrieve("https://raw.githubusercontent.com/darkpegasuslord28-cpu/tolun-farm/main/kaggle_i2v/" + J.get("script", "tolun3d.py"), sc)
    for it in J["shots"]:
        t0 = time.time(); fr = f"/kaggle/working/fr_{it['id']}"; fps = J.get("fps", 30)
        env = dict(os.environ, SCENE=it["scene"], W=str(J.get("w", 1080)), H=str(J.get("h", 1920)), SAMPLES=str(it.get("samples", J.get("samples", 48))), GPU="1")
        r = subprocess.run([sys.executable, sc, "anim", fr, str(fps), str(it["sec"])], env=env, capture_output=True, text=True)
        ok = r.returncode == 0 and os.path.exists(fr)
        if ok:
            subprocess.run([FF, "-y", "-v", "error", "-framerate", str(fps), "-i", f"{fr}/f_%04d.png", "-c:v", "libx264", "-crf", "14", "-preset", "slow", "-pix_fmt", "yuv420p", f"{OUT}/{it['id']}.mp4"])
            fs = sorted(os.listdir(fr)); __import__("shutil").copy(os.path.join(fr, fs[len(fs) // 2]), f"{OUT}/{it['id']}.png")
        log.append({"id": it["id"], "model": "blender", "ok": ok, "sec": round(time.time() - t0), "err": "" if ok else (r.stdout[-600:] + r.stderr[-900:])})
        json.dump(log, open(f"{OUT}/log.json", "w"), indent=1); print(log[-1], flush=True)
    sys.exit(0)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "diffusers>=0.35", "transformers", "accelerate", "sentencepiece", "ftfy", "imageio[ffmpeg]"], check=False)
import torch, numpy as np
from PIL import Image
from diffusers.utils import export_to_video
OUT = "/kaggle/working/out"; os.makedirs(OUT, exist_ok=True); log = []
RAW = "https://raw.githubusercontent.com/darkpegasuslord28-cpu/tolun-farm/artgen-cache/artgen/"
NEG = "static, still image, frozen, no motion, worst quality, blurry, jittery, distorted, text, watermark, people, morphing"
PIPES = {}
def pipe_for(kind):
    if kind in PIPES: return PIPES[kind]
    for k in list(PIPES): del PIPES[k]
    torch.cuda.empty_cache()
    if kind == "wan":
        from diffusers import WanImageToVideoPipeline, AutoencoderKLWan
        mid = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
        vae = AutoencoderKLWan.from_pretrained(mid, subfolder="vae", torch_dtype=torch.float32)
        if torch.cuda.device_count() > 1:   # T4 x2: bileşenleri iki karta doğrudan yükle (CPU RAM tepe yükü yok → "Killed" olmaz)
            p = WanImageToVideoPipeline.from_pretrained(mid, vae=vae, torch_dtype=torch.float16, device_map="balanced"); PIPES[kind] = p; return p
        p = WanImageToVideoPipeline.from_pretrained(mid, vae=vae, torch_dtype=torch.float16, low_cpu_mem_usage=True)
    else:
        from diffusers import LTXImageToVideoPipeline
        p = LTXImageToVideoPipeline.from_pretrained(J.get("ltx_model", "Lightricks/LTX-Video"), torch_dtype=torch.float16)
        p.vae.enable_tiling()
    p.enable_model_cpu_offload(); PIPES[kind] = p; return p
T2I = {}
def t2i(it):   # GÖRSEL ÜRETİMİ (GPU): RealVisXL V5 (SDXL, CreativeML OpenRAIL++-M) — fotogerçekçi ürün çekimleri
    if "p" not in T2I:
        for k in list(PIPES): del PIPES[k]
        torch.cuda.empty_cache()
        from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler
        mid = J.get("t2i_model", "SG161222/RealVisXL_V5.0")
        try: p = StableDiffusionXLPipeline.from_pretrained(mid, torch_dtype=torch.float16, variant="fp16").to("cuda")
        except Exception: p = StableDiffusionXLPipeline.from_pretrained(mid, torch_dtype=torch.float16).to("cuda")
        p.scheduler = DPMSolverMultistepScheduler(num_train_timesteps=1000, beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", algorithm_type="dpmsolver++", solver_order=2, use_karras_sigmas=True, steps_offset=1, timestep_spacing="leading"); p.vae.enable_tiling(); T2I["p"] = p
    w, h = it.get("size", [832, 1216]); g = torch.Generator("cuda").manual_seed(it.get("seed", 7))
    im = T2I["p"](prompt=it["prompt"], negative_prompt=it.get("negative", "text, letters, logo, watermark, people, hands, blurry, lowres, deformed, cartoon, illustration, cgi look"),
                  width=w, height=h, num_inference_steps=it.get("steps", 30), guidance_scale=it.get("cfg", 5.0), generator=g).images[0]
    im.save(f"{OUT}/{it['id']}.png")
def motion(fr):
    a = np.stack([np.asarray(f.convert("L").resize((128, 128)), np.float32) if hasattr(f, "convert") else np.asarray(Image.fromarray((np.asarray(f) * 255).astype(np.uint8)).convert("L").resize((128, 128)), np.float32) for f in fr])
    return round(float(np.abs(np.diff(a, axis=0)).mean()), 3)
for it in J["items"]:
    t0 = time.time(); kind = it.get("model", J.get("model", "ltx"))
    if kind == "t2i":
        try: t2i(it); log.append({"id": it["id"], "model": "t2i", "ok": True, "sec": round(time.time() - t0)})
        except Exception as e: log.append({"id": it["id"], "model": "t2i", "ok": False, "err": repr(e)[:400]})
        json.dump(log, open(f"{OUT}/log.json", "w"), indent=1); print(log[-1], flush=True); continue
    try:
        src = it["image"]; url = src if src.startswith("http") else ("https://raw.githubusercontent.com/darkpegasuslord28-cpu/tolun-farm/i2v-cache/" + src if src.startswith("i2v/") else RAW + src)
        img = Image.open(io.BytesIO(urllib.request.urlopen(url, timeout=120).read())).convert("RGB")
        w, h = it.get("size", J.get("size", [512, 768])); img = img.resize((w, h), Image.LANCZOS)
        p = pipe_for(kind); g = torch.Generator("cuda").manual_seed(it.get("seed", 7))
        print("GPU sayısı", torch.cuda.device_count(), flush=True)
        kw = dict(image=img, prompt=it["prompt"], negative_prompt=NEG, width=w, height=h, num_frames=it.get("frames", J.get("frames", 97)),
                  num_inference_steps=it.get("steps", J.get("steps", 40)), guidance_scale=it.get("cfg", 3.0 if kind == "ltx" else 5.0), generator=g)
        if kind == "ltx": kw.update(decode_timestep=0.03, decode_noise_scale=0.025)
        fr = p(**kw).frames[0]
        export_to_video(fr, f"{OUT}/{it['id']}.mp4", fps=it.get("fps", 24)); log.append({"id": it["id"], "model": kind, "ok": True, "sec": round(time.time() - t0), "motion": motion(fr)})
    except Exception as e:
        log.append({"id": it["id"], "model": kind, "ok": False, "err": repr(e)[:400]})
    json.dump(log, open(f"{OUT}/log.json", "w"), indent=1); print(log[-1], flush=True)
