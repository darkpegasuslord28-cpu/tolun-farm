"""Gece müzik arşivi: library.json'daki duygulardan bugünün grubunu seçer, requests/music/lib-<tarih>-<grup>.json yazar (iş akışı hemen üretir)."""
import json, datetime, os, sys
L = json.load(open("library.json"))["moods"]; day = datetime.date.today(); g = int(sys.argv[1]); n = len(L)
picks = [L[(day.toordinal() * 4 + g + k * 4) % n] for k in range(1)]   # her grup bir duygu
m = picks[0]; name = f"lib-{day:%Y%m%d}-{m['key']}"
tracks = [{"id": f"{m['key']}_{day:%Y%m%d}_{i}", "caption": m["caption"], "duration": 60, "bpm": m["bpm"], "keyscale": m["keyscale"], "seed": day.toordinal() * 10 + i} for i in range(3)]
os.makedirs("requests/music", exist_ok=True); json.dump({"name": name, "tracks": tracks}, open(f"requests/music/{name}.json", "w"), indent=1); print(f"requests/music/{name}.json")
