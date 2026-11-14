"""ÜCRETSİZ SANAT ÜRETİMİ (GitHub Actions CPU): istekteki her sahne için resim (açık model, ticari serbest) + derinlik haritası.
İstek: ARTGEN_REQ = data/artgen/<ad>.json  {"name": "...", "style": "ortak stil tarifi", "negative": "...", "width": 576, "height": 1024, "steps": 6,
        "shots": [{"id": "s1", "prompt": "...", "seed": 7}, ...]}
Çıktı: artgen/<ad>/<id>.png + <id>_depth.png + meta.json  (artgen-cache dalına).
Modeller: Lykon/dreamshaper-8 (SD1.5, CreativeML OpenRAIL-M: ticari serbest) + latent-consistency/lcm-lora-sdv1-5 (az adım) ;
derinlik: depth-anything/Depth-Anything-V2-Small-hf (Apache-2.0)."""
import os, json, time, torch
from diffusers import StableDiffusionPipeline, LCMScheduler
from transformers import pipeline
from PIL import Image
req = json.load(open(os.environ["ARTGEN_REQ"])); name = req["name"]; out = f"artgen/{name}"; os.makedirs(out, exist_ok=True)
torch.set_num_threads(os.cpu_count() or 4)
pipe = StableDiffusionPipeline.from_pretrained(req.get("model", "Lykon/dreamshaper-8"), torch_dtype=torch.float32, safety_checker=None)
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config); pipe.load_lora_weights("latent-consistency/lcm-lora-sdv1-5"); pipe.fuse_lora()
depth = pipeline("depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf")
W, H, steps = req.get("width", 576), req.get("height", 1024), req.get("steps", 6); meta = []
for sh in req["shots"]:
    t = time.time(); g = torch.Generator().manual_seed(sh.get("seed", 1))
    im = pipe(prompt=f'{sh["prompt"]}, {req.get("style", "")}', negative_prompt=req.get("negative", ""), width=W, height=H, num_inference_steps=steps, guidance_scale=req.get("cfg", 1.5), generator=g).images[0]
    im.save(f'{out}/{sh["id"]}.png'); d = depth(im)["depth"]; d.save(f'{out}/{sh["id"]}_depth.png')
    meta.append({"id": sh["id"], "prompt": sh["prompt"], "seed": sh.get("seed", 1), "sec": round(time.time() - t, 1)}); print(sh["id"], round(time.time() - t, 1), "s", flush=True)
json.dump({"name": name, "model": req.get("model", "Lykon/dreamshaper-8"), "license": "CreativeML OpenRAIL-M (model); outputs ours", "shots": meta}, open(f"{out}/meta.json", "w"), indent=1)
