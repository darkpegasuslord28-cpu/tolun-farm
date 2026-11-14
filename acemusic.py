"""ACE-Step 1.5 (MIT lisansı, github.com/ace-step/ACE-Step-1.5) ile ÜCRETSİZ yapay zekâ müziği — GitHub Actions CPU, yalnız DiT (dil modeli kapalı), enstrümantal.
İstek: data/music/req/<ad>.json = {"name": "...", "tracks": [{"id", "caption", "duration", "bpm", "keyscale", "seed", "steps"}]}
Çıktı: music/<ad>/<id>.ogg + <id>.json (prompt, seed, süre, üretim süresi) → music-cache dalı. Atıf: "Music generated with ACE-Step 1.5 (MIT)"."""
import os, sys, json, time, subprocess
REQ = os.environ["MUSIC_REQ"]; ACE = os.environ.get("ACE_DIR", "/tmp/ace"); sys.path.insert(0, ACE)
R = json.load(open(REQ)); out = os.path.join(os.getcwd(), "music", R["name"]); os.makedirs(out, exist_ok=True)
import torch; torch.set_num_threads(os.cpu_count() or 4)
from acestep.handler import AceStepHandler
from acestep.llm_inference import LLMHandler
from acestep.inference import GenerationParams, GenerationConfig, generate_music
dit = AceStepHandler(); t0 = time.time()
msg, ok = dit.initialize_service(project_root=ACE, config_path="acestep-v15-turbo", device="cpu", offload_to_cpu=False, quantization=os.environ.get("ACE_QUANT", "int8_weight_only") or None)
print("init", ok, msg[:300], round(time.time() - t0), "sn"); assert ok, msg
llm = LLMHandler()                                                   # başlatılmadı → DiT-only (CPU'da hızlı yol)
for tr in R["tracks"]:
    t1 = time.time()
    p = GenerationParams(caption=tr["caption"], lyrics="[Instrumental]", instrumental=True, duration=float(tr.get("duration", 60)), bpm=tr.get("bpm"),
                         keyscale=tr.get("keyscale", ""), inference_steps=int(tr.get("steps", 8)), seed=int(tr.get("seed", 42)), thinking=False)
    res = generate_music(dit, llm, p, GenerationConfig(batch_size=1, audio_format="wav"), save_dir=out)
    if not res.success: print("HATA", tr["id"], res.error); continue
    src = res.audios[0]["path"]; dst = os.path.join(out, tr["id"] + ".ogg")
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", src, "-c:a", "libvorbis", "-q:a", "5", dst], check=True); os.remove(src)
    meta = dict(tr, gen_seconds=round(time.time() - t1), model="ACE-Step 1.5 turbo (MIT)", credit="Music generated with ACE-Step 1.5 (MIT License)")
    json.dump(meta, open(os.path.join(out, tr["id"] + ".json"), "w"), ensure_ascii=False, indent=1); print("OK", tr["id"], meta["gen_seconds"], "sn")
