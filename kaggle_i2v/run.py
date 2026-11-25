"""KAGGLE GPU İŞİ (tolun-farm i2v): job.json'daki her sabit görseli GERÇEK hareketli klibe çevirir (görüntü→video).
Modeller: "ltx" = Lightricks LTX-Video (diffusers LTXImageToVideoPipeline) · "wan" = Wan2.2-TI2V-5B (Apache-2.0, diffusers).
Çıktı: /kaggle/working/out/<id>.mp4 (+ log.json, hareket ölçüsü). Görseller tolun-farm artgen-cache'ten. Kişisel bilgi/anahtar YOK."""
import json, os, sys, time, subprocess, urllib.request, io
def torch_cuda_n():
    import torch; return torch.cuda.device_count()
J = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "job.json")))
# --- HIZLANDIRMA (19 Kas): hazır ortam + tek oturumda çok iş ---------------------------------------------
if os.environ.get("SUBJOB"): J = json.load(open(os.environ["SUBJOB"]))
OUT_ROOT = os.environ.get("OUT_ROOT", "/kaggle/working/out")
def _find_env():   # Kaggle çekirdek çıktısını farklı yollara bağlayabiliyor: /kaggle/input altında 'whisper' klasörü olan dizini ara
    for root, dirs, _ in os.walk("/kaggle/input"):
        if root.count(os.sep) > 6: dirs[:] = []; continue
        if "whisper" in dirs and "models" in dirs: return root
    return None
def M(rid):   # hazır ortamda yerel kopya varsa onu kullan, yoksa HF kimliği
    d = os.path.join(os.environ.get("MODELS_DIR", "/nonexistent"), rid.replace("/", "__"))
    return d if os.path.isdir(d) else rid
ENV = _find_env() if os.path.isdir("/kaggle/input") and not os.environ.get("ENV_READY") else None
if os.path.isdir("/kaggle/input") and not ENV and not os.environ.get("ENV_READY"): print("hazır ortam yok; /kaggle/input:", os.listdir("/kaggle/input"), flush=True)
if ENV and not os.environ.get("ENV_READY"):   # model önbelleği: salt-okunur girdiden sembolik bağ ağacı (kilit dosyaları yazılabilsin)
    t0 = time.time()
    if os.path.isdir(ENV + "/models"): os.environ["MODELS_DIR"] = ENV + "/models"   # model klasörleri (sembolik bağsız, Kaggle çıktısı bağları korumaz)
    if os.path.isdir(ENV + "/whisper"): os.environ["WHISPER_ROOT"] = ENV + "/whisper"
    os.environ["ENV_READY"] = "1"; print("hazır ortam bağlandı", ENV, round(time.time() - t0, 1), "sn", flush=True)
if J.get("mode") == "env":   # HAZIR ORTAM ÇEKİRDEĞİ: ağırlıkları bir kez indir → çıktısı sonraki işlerde kernel_sources ile girdi olur
    W = "/kaggle/working"; t0 = time.time()
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub", "openai-whisper"], check=False)
    from huggingface_hub import snapshot_download
    for rid in J.get("hf", []):
        try: snapshot_download(rid, allow_patterns=J.get("patterns", {}).get(rid), local_dir=f"{W}/models/" + rid.replace("/", "__")); print("indi", rid, round(time.time() - t0), flush=True)
        except Exception as e: print("inemedi", rid, repr(e)[:200], flush=True)
    import whisper; whisper._download(whisper._MODELS[J.get("asr", "large-v3-turbo")], W + "/whisper", False); print("whisper indi", flush=True)
    subprocess.run("du -sh /kaggle/working/models/* /kaggle/working/whisper", shell=True); sys.exit(0)
if J.get("jobs"):   # TEK OTURUM, ÇOK İŞ: alt işler sırayla (ya da "parallel": GPU başına bir iş, T4×2)
    import torch; ng = max(1, torch.cuda.device_count()) if J.get("parallel") else 1; subs = J["jobs"]; running = []
    if ng > 1:   # paralel alt işler aynı anda pip'e yazmasın: paketleri önce bir kez kur
        t0 = time.time(); subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "voxcpm", "openai-whisper", "jiwer", "librosa", "diffusers>=0.35", "transformers", "accelerate", "sentencepiece", "ftfy", "imageio[ffmpeg]"], check=False); print("paketler", round(time.time() - t0), "sn", flush=True)
    for k, sub in enumerate(subs):
        jp = f"/kaggle/working/sub_{k}.json"; json.dump(sub, open(jp, "w"))
        env = dict(os.environ, SUBJOB=jp, OUT_ROOT=f"/kaggle/working/out/{sub['name']}")
        if ng > 1: env["CUDA_VISIBLE_DEVICES"] = str(k % ng)
        while len(running) >= ng: running = [p for p in running if p.poll() is None]; time.sleep(5)
        print("alt iş", sub["name"], "GPU", env.get("CUDA_VISIBLE_DEVICES", "hepsi"), flush=True)
        running.append(subprocess.Popen([sys.executable, os.path.abspath(__file__)], env=env))
        if ng == 1: running[-1].wait()
    for p in running: p.wait()
    sys.exit(0)
if J.get("mode") == "chatterbox":   # TÜRKÇE İFADELİ SES (GPU): Chatterbox Multilingual (Resemble AI, MIT) — referans sesten klon + exaggeration/cfg
    OUT = OUT_ROOT; os.makedirs(OUT, exist_ok=True); log = []
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "chatterbox-tts", "transformers==4.46.3", "diffusers==0.29.0"], check=False)
    import torch, torchaudio
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS
    m = ChatterboxMultilingualTTS.from_pretrained(device="cuda")
    refs = {}
    for v, src in J.get("refs", {}).items():
        url = src if src.startswith("http") else "https://raw.githubusercontent.com/darkpegasuslord28-cpu/tolun-farm/i2v-cache/" + src
        refs[v] = f"/kaggle/working/ref_{v}.wav"; urllib.request.urlretrieve(url, refs[v])
    for it in J["lines"]:
        for k in range(J.get("takes", 2)):
            t0 = time.time(); oid = f"{it['id']}_t{k}"
            try:
                torch.manual_seed(J.get("seed", 1) + k)
                wav = m.generate(it["text"], language_id="tr", audio_prompt_path=refs.get(it.get("ref")), exaggeration=it.get("exaggeration", 0.7), cfg_weight=it.get("cfg", 0.3))
                torchaudio.save(f"{OUT}/{oid}.wav", wav.cpu(), m.sr); log.append({"id": oid, "ok": True, "sec": round(time.time() - t0)})
            except Exception as e:
                log.append({"id": oid, "ok": False, "err": repr(e)[:400]})
            json.dump(log, open(f"{OUT}/log.json", "w"), indent=1); print(log[-1], flush=True)
    sys.exit(0)
if J.get("mode") == "voxcpm":   # TÜRKÇE SESLENDİRME (GPU): VoxCPM2 (OpenBMB, Apache-2.0) ses tasarımı/klon → her replik N deneme → Whisper ile doğruluk → en iyisi
    OUT = OUT_ROOT; os.makedirs(OUT, exist_ok=True); log = []
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "voxcpm", "openai-whisper", "jiwer"], check=False)
    import soundfile as sf, whisper, jiwer, re, unicodedata, numpy as np
    from voxcpm import VoxCPM
    m = None
    for mid_ in ([J["tts_model"]] if J.get("tts_model") else []) + ["niobures/Trendyol-TTS", "openbmb/VoxCPM2"] if "rendyol" in J.get("tts_model", "") else [J.get("tts_model", "openbmb/VoxCPM2")]:
        try: m = VoxCPM.from_pretrained(M(mid_), load_denoiser=False); print("model:", mid_, flush=True); break
        except Exception as e: print("model yüklenemedi", mid_, repr(e)[:200], flush=True)
    SR = m.tts_model.sample_rate
    asr = whisper.load_model(J.get("asr", "large-v3-turbo"), download_root=os.environ.get("WHISPER_ROOT"), device="cuda:1" if torch_cuda_n() > 1 else "cuda")
    def norm(t): t = unicodedata.normalize("NFC", t.lower()); return re.sub(r"[^\wçğıöşü ]", " ", t).split()
    REFS = {}
    for v, V in J["voices"].items():
        if V.get("ref_url"): rp = f"/tmp/ref_{v}.wav"; urllib.request.urlretrieve(V["ref_url"], rp); REFS[v] = rp
    REF_TR = "Akşam olmuştu ve okulun uzun koridorları yavaş yavaş sessizleşiyordu. Kimse ne olacağını bilmiyordu, ama herkes bir şeylerin değişmek üzere olduğunu hissediyordu."
    for v, V in J["voices"].items():          # tasarlanan sesi BİR KEZ üret → tüm repliklerde klon referansı (karakter sesi sabit kalır)
        if v in REFS or not V.get("design"): continue
        bestr = None
        for k in range(4):
            w = m.generate(text=f"({V['design']})" + REF_TR, cfg_value=V.get("cfg", 2.0), inference_timesteps=J.get("steps", 12)); fn = f"{OUT}/ref_{v}_t{k}.wav"; sf.write(fn, w, SR)
            hyp = asr.transcribe(fn, language="tr", fp16=True)["text"]; er = jiwer.cer(" ".join(norm(REF_TR)), " ".join(norm(hyp)) or "-")
            if bestr is None or er < bestr[0]: bestr = (er, fn)
        os.system(f"cp {bestr[1]} {OUT}/ref_{v}.wav"); REFS[v] = f"{OUT}/ref_{v}.wav"; print("ref", v, round(bestr[0], 3), flush=True)
    for L in J["lines"]:
        V = J["voices"][L["voice"]]; t0 = time.time(); best = None
        for k in range(J.get("takes", 3)):
            txt = (f"({V['design']}" + (f", {L['style']}" if L.get("style") else "") + ")" if V.get("design") and L["voice"] not in REFS else (f"({L['style']})" if L.get("style") else "")) + L["text"]
            kw = dict(text=txt, cfg_value=V.get("cfg", 2.0), inference_timesteps=J.get("steps", 12))
            if L["voice"] in REFS: kw["reference_wav_path"] = REFS[L["voice"]]
            try:
                w = m.generate(**kw); fn = f"{OUT}/{L['id']}_t{k}.wav"; sf.write(fn, w, SR)
                hyp = asr.transcribe(fn, language="tr", fp16=True)["text"]; er = jiwer.cer(" ".join(norm(L["text"])), " ".join(norm(hyp)) or "-")
                dur = len(w) / SR; cps = len(L["text"]) / max(dur, 0.1); pen = 0 if 8 <= cps <= 17 else 0.05
                try:                                                  # ifade gücü: perde + enerji değişkenliği (duygulu okuma ödüllendirilir)
                    import librosa; y16 = librosa.resample(np.asarray(w, dtype=np.float32), orig_sr=SR, target_sr=16000)
                    f0, _, _ = librosa.pyin(y16, fmin=60, fmax=500, sr=16000); f0 = f0[~np.isnan(f0)]
                    rms = librosa.feature.rms(y=y16)[0]; ex = (np.std(np.log(f0)) if len(f0) > 10 else 0) * 4 + np.std(rms) / (np.mean(rms) + 1e-6) * 0.5
                    # cümle sonu düşüşü: "." ya da "!" ile biten replikte son perde, cümlenin ilk yarısından ≥2 yarım ton aşağıda olmalı (yoksa "devam edecek gibi")
                    fall = 12 * np.log2(np.median(f0[-12:]) / np.median(f0[:max(1, len(f0) // 2)])) if len(f0) > 24 else -3
                except Exception: ex, fall = 0, -3
                endq = L["text"].rstrip().endswith(("?", "...", "…"))
                sc = er + pen - 0.06 * min(1.5, ex) * (1 if er < 0.12 else 0) + (0 if endq or fall <= -2 else 0.04)
                if best is None or sc < best[0]: best = (sc, fn, hyp, round(dur, 2))
            except Exception as e: print("hata", L["id"], repr(e)[:200], flush=True)
        if best:
            os.system(f"cp {best[1]} {OUT}/{L['id']}.wav")
            try:
                wr = asr.transcribe(best[1], language="tr", fp16=True, word_timestamps=True)
                json.dump([{"w": x["word"].strip(), "s": round(x["start"], 3), "e": round(x["end"], 3)} for sg in wr["segments"] for x in sg.get("words", [])], open(f"{OUT}/{L['id']}.words.json", "w"), ensure_ascii=False)
            except Exception as e: print("kelime zamanı hatası", repr(e)[:150])
        log.append({"id": L["id"], "ok": bool(best), "cer": float(round(float(best[0]), 3)) if best else None, "asr": best[2] if best else "", "dur": best[3] if best else 0, "sec": round(time.time() - t0)})
        print(log[-1], flush=True); json.dump(log, open(f"{OUT}/log.json", "w"), ensure_ascii=False, indent=1)
    sys.exit(0)
if J.get("mode") == "plate":   # FOTOĞRAF PLAKASI → 3B KAMERA (matte painting): görsel + derinlik (Depth Anything V2) → Blender projeksiyon, GPU
    OUT = OUT_ROOT; os.makedirs(OUT, exist_ok=True); log = []
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "bpy", "imageio-ffmpeg", "transformers"], check=False)
    import imageio_ffmpeg; FF = imageio_ffmpeg.get_ffmpeg_exe()
    sc = "/kaggle/working/plate3d.py"; urllib.request.urlretrieve("https://raw.githubusercontent.com/darkpegasuslord28-cpu/tolun-farm/main/kaggle_i2v/plate3d.py", sc)
    from transformers import pipeline as _pl; from PIL import Image as _I; import numpy as _np
    dpipe = _pl("depth-estimation", model="depth-anything/Depth-Anything-V2-Large-hf", device=0)
    def fetch(src, dst):
        url = src if src.startswith("http") else "https://raw.githubusercontent.com/darkpegasuslord28-cpu/tolun-farm/i2v-cache/" + src
        urllib.request.urlretrieve(url, dst); return dst
    done = {}
    for it in J["items"]:
        t0 = time.time()
        try:
            img = done.get(it["image"]) or fetch(it["image"], f"/kaggle/working/p_{len(done)}.png")
            dep = img[:-4] + ".depth.png"
            if not os.path.exists(dep):
                im = _I.open(img).convert("RGB"); d = _np.array(dpipe(im)["predicted_depth"].squeeze().cpu(), dtype=_np.float32)
                d = (d - d.min()) / (d.max() - d.min() + 1e-6); _I.fromarray((d * 65535).astype(_np.uint16)).resize(im.size, _I.BICUBIC).save(dep)
                __import__("shutil").copy(dep, f"{OUT}/{it['id']}.depth.png")
            done[it["image"]] = img
            cmd = [sys.executable, sc, img, dep, f"{OUT}/{it['id']}.mp4", "--move", it.get("move", "push"), "--sec", str(it.get("sec", 5)), "--w", str(J.get("w", 1080)), "--h", str(J.get("h", 1920)), "--amt", str(it.get("amt", 1.0)), "--grid", str(J.get("grid", 600))]
            for k in ("eyes", "blink", "flare"):
                if it.get(k): cmd += ["--" + k, it[k]]
            r = subprocess.run(cmd, env=dict(os.environ, GPU="1", FFMPEG=FF), capture_output=True, text=True)
            log.append({"id": it["id"], "model": "plate", "ok": r.returncode == 0, "sec": round(time.time() - t0), "err": "" if r.returncode == 0 else (r.stdout[-500:] + r.stderr[-800:])})
        except Exception as e:
            log.append({"id": it["id"], "model": "plate", "ok": False, "err": repr(e)[:400]})
        json.dump(log, open(f"{OUT}/log.json", "w"), indent=1); print(log[-1], flush=True)
    sys.exit(0)
if J.get("mode") == "blender":   # 3B ÜRÜN SAHNELERİ (Blender/bpy, Cycles GPU): kaggle_i2v/tolun3d.py — her sahnede aynı ürün
    OUT = OUT_ROOT; os.makedirs(OUT, exist_ok=True); log = []
    subprocess.run("apt-get install -y -qq libxi6 libxxf86vm1 libxfixes3 libxrender1 libgl1 libxkbcommon0 libsm6 > /dev/null 2>&1", shell=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "bpy", "imageio-ffmpeg"], check=False)
    import imageio_ffmpeg; FF = imageio_ffmpeg.get_ffmpeg_exe()
    sc = "/kaggle/working/" + J.get("script", "tolun3d.py")
    urllib.request.urlretrieve("https://raw.githubusercontent.com/darkpegasuslord28-cpu/tolun-farm/main/kaggle_i2v/" + J.get("script", "tolun3d.py"), sc)
    for it in J["shots"]:
        t0 = time.time(); fr = f"/kaggle/working/fr_{it['id']}"; fps = J.get("fps", 30)
        env = dict(os.environ, HDRI=J.get("hdri", ""), SCENE=it["scene"], W=str(J.get("w", 1080)), H=str(J.get("h", 1920)), SAMPLES=str(it.get("samples", J.get("samples", 48))), GPU="1")
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
OUT = OUT_ROOT; os.makedirs(OUT, exist_ok=True); log = []
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
        sch = DPMSolverMultistepScheduler(num_train_timesteps=1000, beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", algorithm_type="dpmsolver++", solver_order=2, use_karras_sigmas=True, steps_offset=1, timestep_spacing="leading")
        try: p = StableDiffusionXLPipeline.from_pretrained(M(mid), torch_dtype=torch.float16, variant="fp16", scheduler=sch).to("cuda")
        except Exception as e: print("fp16 varyant yok:", repr(e)[:200], flush=True); p = StableDiffusionXLPipeline.from_pretrained(M(mid), torch_dtype=torch.float16, scheduler=sch).to("cuda")
        p.scheduler = DPMSolverMultistepScheduler(num_train_timesteps=1000, beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", algorithm_type="dpmsolver++", solver_order=2, use_karras_sigmas=True, steps_offset=1, timestep_spacing="leading"); p.vae.enable_tiling()
        if J.get("ip_adapter"):   # karakter tutarlılığı: IP-Adapter Plus Face (h94, Apache-2.0) — referans yüz kartı
            p.load_ip_adapter(M("h94/IP-Adapter"), subfolder="sdxl_models", weight_name="ip-adapter-plus-face_sdxl_vit-h.safetensors", image_encoder_folder="models/image_encoder")
        T2I["p"] = p
    w, h = it.get("size", [832, 1216]); g = torch.Generator("cuda").manual_seed(it.get("seed", 7))
    kw = {}
    if J.get("ip_adapter"):
        from PIL import Image as _I
        if it.get("ref"):
            rp = f"/tmp/ref_{os.path.basename(it['ref'])}"
            if not os.path.exists(rp): urllib.request.urlretrieve("https://raw.githubusercontent.com/darkpegasuslord28-cpu/tolun-farm/i2v-cache/" + it["ref"], rp)
            kw["ip_adapter_image"] = _I.open(rp).convert("RGB"); T2I["p"].set_ip_adapter_scale(it.get("ip_scale", 0.55))
        else:
            kw["ip_adapter_image"] = _I.new("RGB", (224, 224)); T2I["p"].set_ip_adapter_scale(0.0)
    ims = []
    for ci in range(it.get("cands", 1)):   # SAHNE = REPLİK: birkaç aday üret, CLIP ile replikteki eyleme en çok benzeyeni seç
        g = torch.Generator("cuda").manual_seed(it.get("seed", 7) + ci * 17)
        ims.append(T2I["p"](**kw, prompt=it["prompt"], negative_prompt=it.get("negative", "text, letters, logo, watermark, people, hands, blurry, lowres, deformed, cartoon, illustration, cgi look"),
                  width=w, height=h, num_inference_steps=it.get("steps", 30), guidance_scale=it.get("cfg", 5.0), generator=g).images[0])
    sc = clip_pick(ims, it.get("action") or it["prompt"].split(",")[0]) if len(ims) > 1 else [0.0]
    best = int(np.argmax(sc)); ims[best].save(f"{OUT}/{it['id']}.png"); it["_clip"] = [round(float(x), 3) for x in sc]
    for ci, im_ in enumerate(ims):
        if ci != best: im_.resize((im_.width // 3, im_.height // 3)).save(f"{OUT}/{it['id']}_alt{ci}.jpg")
CLIP = {}
def clip_pick(ims, text):   # openai/clip-vit-large-patch14 (MIT): görüntü ↔ eylem cümlesi benzerliği
    from transformers import CLIPModel, CLIPProcessor
    if "m" not in CLIP: CLIP["m"] = CLIPModel.from_pretrained(M("openai/clip-vit-large-patch14"), torch_dtype=torch.float16).to("cuda"); CLIP["p"] = CLIPProcessor.from_pretrained(M("openai/clip-vit-large-patch14"))
    x = CLIP["p"](text=[text], images=ims, return_tensors="pt", padding=True, truncation=True).to("cuda")
    with torch.no_grad(): o = CLIP["m"](**{k: (v.half() if v.dtype == torch.float32 else v) for k, v in x.items()})
    return o.logits_per_image[:, 0].float().cpu().numpy()
def motion(fr):
    a = np.stack([np.asarray(f.convert("L").resize((128, 128)), np.float32) if hasattr(f, "convert") else np.asarray(Image.fromarray((np.asarray(f) * 255).astype(np.uint8)).convert("L").resize((128, 128)), np.float32) for f in fr])
    return round(float(np.abs(np.diff(a, axis=0)).mean()), 3)
for it in J["items"]:
    t0 = time.time(); kind = it.get("model", J.get("model", "ltx"))
    if kind == "t2i":
        try: t2i(it); log.append({"id": it["id"], "model": "t2i", "ok": True, "sec": round(time.time() - t0), "clip": it.get("_clip")})
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
