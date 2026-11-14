# tolun-farm

Free, open-source generation workers for **Tolun Studio** channels, running on public GitHub Actions CPU runners.

- `requests/art/<name>.json` → paintings + depth maps (DreamShaper-8 SD1.5, CreativeML OpenRAIL-M; LCM-LoRA; Depth-Anything-V2-Small, Apache-2.0) → branch `artgen-cache` (`artgen/<name>/`).
- `requests/music/<name>.json` → instrumental music (ACE-Step 1.5, MIT) → branch `music-cache` (`music/<name>/`).

Push a request file to `main` and the matching workflow starts automatically (one parallel job per request). No secrets live here; outputs are original generated media.
