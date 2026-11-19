"""ÇOK SESLİ TÜRKÇE SESLENDİRME (tolun-farm, CPU): Chatterbox Multilingual (Resemble AI, MIT) — sıfır-atış ses klonlama + duygu ayarı.
İstek: requests/tts/<ad>.json  {"name", "lines": [{"id","voice","text","exag"?}], "voices": {ad: {"ref": "default"|"piper:<model>"|"url:<https wav>", "pitch": yarım ton, "exag", "cfg"}}}
Referans sesler: Piper Türkçe modelleri (rhasspy/piper-voices) ya da Chatterbox'ın kendi sesi; perde kaydırmayla farklı karakterler.
Gerçek kişi sesi (izinli) ileride "url:" ile verilir — bu repo PUBLIC, kişisel ses dosyası buraya KONMAZ.
Çıktı: tts/<ad>/<id>.wav + log.json → tts-cache dalı."""
import json, os, sys, subprocess, wave, urllib.request, time, random
import torch
_l = torch.load; torch.load = lambda *a, **k: _l(*a, **{**k, "map_location": k.get("map_location", "cpu")})   # CPU'da yükle
import torchaudio as ta
from chatterbox.mtl_tts import ChatterboxMultilingualTTS
J = json.load(open(os.environ["TTS_REQ"])); NAME = J["name"]; OUT = f"tts/{NAME}"; os.makedirs(OUT, exist_ok=True); os.makedirs("/tmp/ref", exist_ok=True)
REF_TXT = ("Akşam olmuştu ve okulun koridorları yavaş yavaş sessizleşiyordu. Pencerelerden soğuk bir rüzgâr giriyordu. "
           "Kimse ne olacağını bilmiyordu, ama herkes bir şeylerin değişmek üzere olduğunu hissediyordu.")
PIPER = "https://huggingface.co/rhasspy/piper-voices/resolve/main/{lang}/{loc}/{spk}/{q}/{loc}-{spk}-{q}.onnx"
REF_EN = ("It was getting dark, and the long hallways of the old school were slowly going quiet. A cold wind came through the windows. "
          "Nobody knew what was about to happen, but everyone felt that something was about to change.")
def shift(src, dst, st):
    if abs(st) < 0.01: subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", src, "-ar", "24000", "-ac", "1", dst], check=True); return
    k = 2 ** (st / 12); r = subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", src, "-af", f"rubberband=pitch={k:.4f}", "-ar", "24000", "-ac", "1", dst])
    if r.returncode: subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", src, "-af", f"asetrate=24000*{k:.4f},aresample=24000,atempo={1 / k:.4f}", "-ar", "24000", "-ac", "1", dst], check=True)
def piper_ref(model, dst):
    loc, spk, q = model.split("-")[0], model.split("-")[1], model.split("-")[2]
    p = f"/tmp/ref/{model}.onnx"
    if not os.path.exists(p):
        u = PIPER.format(lang=loc.split("_")[0], loc=loc, spk=spk, q=q); urllib.request.urlretrieve(u, p); urllib.request.urlretrieve(u + ".json", p + ".json")
    txt = REF_TXT if loc.startswith("tr") else REF_EN   # yabancı model yalnız ses rengi (tını) referansı; konuşma yine Türkçe üretilir
    from piper import PiperVoice
    v = PiperVoice.load(p)
    with wave.open(dst, "wb") as wf:
        if hasattr(v, "synthesize_wav"): v.synthesize_wav(txt, wf)
        else: v.synthesize(txt, wf)
model = ChatterboxMultilingualTTS.from_pretrained(device="cpu")
REF = {}; log = []
for name, V in J["voices"].items():
    ref = V.get("ref", "default"); raw = f"/tmp/ref/{name}_raw.wav"
    try:
        if ref == "default":
            if abs(V.get("pitch", 0)) < 0.01: REF[name] = None; continue
            w = model.generate(REF_TXT, language_id="tr"); ta.save(raw, w, model.sr)
        elif ref.startswith("piper:"): piper_ref(ref[6:], raw)
        elif ref.startswith("url:"): urllib.request.urlretrieve(ref[4:], raw)
        dst = f"/tmp/ref/{name}.wav"; shift(raw, dst, V.get("pitch", 0)); REF[name] = dst
        os.system(f"cp {dst} {OUT}/ref_{name}.wav")
    except Exception as e:
        print("REF HATA", name, repr(e)); REF[name] = None
for L in J["lines"]:
    V = J["voices"].get(L["voice"], {}); t0 = time.time(); torch.manual_seed(L.get("seed", 1234)); random.seed(1234)
    try:
        kw = dict(language_id="tr", exaggeration=L.get("exag", V.get("exag", 0.5)), cfg_weight=L.get("cfg", V.get("cfg", 0.5)))
        if REF.get(L["voice"]): kw["audio_prompt_path"] = REF[L["voice"]]
        w = model.generate(L["text"], **kw); ta.save(f"{OUT}/{L['id']}.wav", w, model.sr)
        log.append({"id": L["id"], "voice": L["voice"], "ok": True, "sec": round(time.time() - t0, 1), "dur": round(w.shape[-1] / model.sr, 2)})
    except Exception as e:
        log.append({"id": L["id"], "ok": False, "err": repr(e)[:300]})
    print(log[-1], flush=True); json.dump(log, open(f"{OUT}/log.json", "w"), indent=1)
