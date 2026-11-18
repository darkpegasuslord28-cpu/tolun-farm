# tolun-farm

Free, open-source generation workers for **Tolun Studio** channels, running on public GitHub Actions CPU runners.

- `requests/art/<name>.json` → paintings + depth maps (DreamShaper-8 SD1.5, CreativeML OpenRAIL-M; LCM-LoRA; Depth-Anything-V2-Small, Apache-2.0) → branch `artgen-cache` (`artgen/<name>/`).
- `requests/music/<name>.json` → instrumental music (ACE-Step 1.5, MIT) → branch `music-cache` (`music/<name>/`).

Push a request file to `main` and the matching workflow starts automatically (one parallel job per request). No secrets live here; outputs are original generated media.

## İkinci çiftlik: Kaggle GPU (görüntü → video)
`requests/i2v/<ad>.json` → `.github/workflows/i2v.yml` Kaggle'da (ücretsiz T4 GPU, haftada ~30 saat) `kaggle_i2v/run.py` çalıştırır → `i2v-cache` dalına mp4.
Secrets (bu repoda, Oğulcan girer): `KAGGLE_USERNAME`, `KAGGLE_KEY`. Model: Lightricks LTX-Video (diffusers). Görseller artgen-cache'ten.
