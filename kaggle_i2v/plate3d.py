"""PLAKA → 3B KAMERA (matte painting / kamera projeksiyonu, 25 Kas 2026).
Fotoğraf gerçekliğindeki görsel + derinlik haritası → derinliğe göre bükülmüş 3B yüzey → içinde gerçek kamera hareketi.
Görüntü fotoğraf kalitesinde kalır, paralaks gerçek 3B hissi verir. Işık yok (emisyon) → CPU'da bile hızlı.
Kullanım:
  python3 tools/plate3d.py <görsel.png> <derinlik.png> <çıktı.mp4> --move push|pan_l|pan_r|look_l|look_r|orbit|walk|rise|dolly_out --sec 5 [--w 1080 --h 1920]
     [--eyes x1,y1,x2,y2]  (görsel pikselinde göz merkezleri → 3B'de parlayan, kırpan gözler)  [--blink 0.3,0.7] [--flare 0.85]
Derinlik: tools/depth.py (Depth Anything V2, açık = yakın)."""
import bpy, sys, os, math, argparse, subprocess, shutil
import numpy as np
argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
ap = argparse.ArgumentParser(); ap.add_argument("img"); ap.add_argument("depth"); ap.add_argument("out")
ap.add_argument("--move", default="push"); ap.add_argument("--sec", type=float, default=5); ap.add_argument("--fps", type=int, default=30)
ap.add_argument("--w", type=int, default=1080); ap.add_argument("--h", type=int, default=1920); ap.add_argument("--amt", type=float, default=1.0)
ap.add_argument("--eyes", default=""); ap.add_argument("--blink", default=""); ap.add_argument("--flare", default=""); ap.add_argument("--grid", type=int, default=400)
ap.add_argument("--still", type=float, default=-1)
A = ap.parse_args(argv)
from PIL import Image
im = Image.open(A.img).convert("RGB"); IW, IH = im.size
dp = np.asarray(Image.open(A.depth).convert("I;16") if Image.open(A.depth).mode.startswith("I") else Image.open(A.depth).convert("L"), np.float32)
dp = np.asarray(Image.fromarray(dp).resize((IW, IH)), np.float32); dp = (dp - dp.min()) / (dp.max() - dp.min() + 1e-6)
# açık=yakın → mesafe: yakın 1.0, uzak 6.0 (ters orantı daha doğal)
Z = 1.0 / (0.17 + 0.83 * dp) ; Z = np.clip(Z, 1.0, 6.0)
bpy.ops.wm.read_factory_settings(use_empty=True); S = bpy.context.scene
N = int(round(A.sec * A.fps)); S.frame_start, S.frame_end = 1, N; S.render.fps = A.fps
FOV = 2 * math.atan(0.5 * 36 / 30)   # yaklaşık 30 mm (36 mm film) yatay açı — orijinal çekim
aspect = IH / IW
# ızgara: (gx+1)×(gy+1) köşe
gx = A.grid; gy = int(round(gx * aspect))
us = np.linspace(0, 1, gx + 1); vs = np.linspace(0, 1, gy + 1)
verts, uvs = [], []
tanx = math.tan(FOV / 2); tany = tanx * aspect
for j, v in enumerate(vs):
    for i, u in enumerate(us):
        px = min(IW - 1, int(u * (IW - 1))); py = min(IH - 1, int(v * (IH - 1))); z = float(Z[py, px])
        x = (u * 2 - 1) * tanx * z; y = (1 - v * 2) * tany * z
        verts.append((x, z, y))   # kamera +y'ye bakar
faces = []
for j in range(gy):
    for i in range(gx):
        a = j * (gx + 1) + i; faces.append((a, a + 1, a + gx + 2, a + gx + 1))
me = bpy.data.meshes.new("plate"); me.from_pydata(verts, [], faces); me.update()
uvl = me.uv_layers.new();
for poly in me.polygons:
    for li in poly.loop_indices:
        vi = me.loops[li].vertex_index; i = vi % (gx + 1); j = vi // (gx + 1); uvl.data[li].uv = (us[i], 1 - vs[j])
ob = bpy.data.objects.new("plate", me); S.collection.objects.link(ob)
# arka perde: aynı görselin uzak kopyası (delikleri doldurur)
def emit_img(name, path):
    m = bpy.data.materials.new(name); m.use_nodes = True; nt = m.node_tree; nt.nodes.clear()
    t = nt.nodes.new("ShaderNodeTexImage"); t.image = bpy.data.images.load(os.path.abspath(path)); t.extension = "EXTEND"
    e = nt.nodes.new("ShaderNodeEmission"); nt.links.new(t.outputs[0], e.inputs[0]); nt.links.new(e.outputs[0], nt.nodes.new("ShaderNodeOutputMaterial").inputs[0]); return m
M = emit_img("plate", A.img); me.materials.append(M)
bpy.ops.mesh.primitive_plane_add(size=1, location=(0, 6.2, 0), rotation=(math.pi / 2, 0, 0)); bg = bpy.context.object
bg.scale = (tanx * 6.2 * 2.3, tany * 6.2 * 2.3, 1); bg.data.materials.append(M)
# kamera
cd = bpy.data.cameras.new("c"); cam = bpy.data.objects.new("cam", cd); S.collection.objects.link(cam); S.camera = cam
cd.sensor_fit = "HORIZONTAL"; cd.sensor_width = 36; cd.lens = 30 * 1.06   # orijinal açıyla eş (+%6 kenar payı)
if (A.h / A.w) > aspect * 1.1: cd.lens = 30 * (A.h / A.w) / aspect * 1.06   # görsel yataysa dikey pencere: içinden kırp
cam.rotation_mode = "XYZ"
def K(t, loc, rot):   # rot: (pitch, roll, yaw) derece; kamera +y'ye bakar
    cam.location = loc; cam.rotation_euler = (math.radians(90 + rot[0]), math.radians(rot[1]), math.radians(rot[2])); f = 1 + int(round(t * (N - 1)))
    cam.keyframe_insert("location", frame=f); cam.keyframe_insert("rotation_euler", frame=f)
a = A.amt
moves = {
    "push":      [(0, (0, 0, 0), (0, 0, 0)), (1, (0, 0.35 * a, 0.01), (0, 0, 0))],
    "dolly_out": [(0, (0, 0.6 * a, 0), (0, 0, 0)), (1, (0, -0.05, 0), (0, 0, 0))],
    "pan_l":     [(0, (0.25 * a, 0.1, 0), (0, 0, 0)), (1, (-0.25 * a, 0.1, 0), (0, 0, 0))],
    "pan_r":     [(0, (-0.25 * a, 0.1, 0), (0, 0, 0)), (1, (0.25 * a, 0.1, 0), (0, 0, 0))],
    "look_l":    [(0, (0, 0.15, 0), (0, 0, -6 * a)), (0.5, (0, 0.2, 0), (0, 0, 2)), (1, (0.02, 0.25, 0), (0, 0, 7 * a))],
    "look_r":    [(0, (0, 0.15, 0), (0, 0, 6 * a)), (0.5, (0, 0.2, 0), (0, 0, -2)), (1, (-0.02, 0.25, 0), (0, 0, -7 * a))],
    "orbit":     [(0, (-0.18 * a, 0.12, 0.01), (0, 0, -4 * a)), (1, (0.18 * a, 0.22, 0.01), (0, 0, 4 * a))],
    "walk":      [(0, (0, 0, 0), (0, 0, 0))] + [(k / 8, (0.006 * math.sin(k * 1.6), 0.045 * k * a, 0.008 * abs(math.sin(k * 1.6))), (0.3 * math.sin(k * 0.8), 0.25 * math.sin(k * 1.6), 0)) for k in range(1, 9)],
    "rise":      [(0, (0, 0.1, -0.2 * a), (4, 0, 0)), (1, (0, 0.3, 0.12 * a), (-3, 0, 0))],
}
for t, loc, rot in moves[A.move]: K(t, loc, rot)
if cam.animation_data and cam.animation_data.action and hasattr(cam.animation_data.action, "fcurves"):
    for fc in cam.animation_data.action.fcurves:
        for kp in fc.keyframe_points: kp.interpolation = "BEZIER"; kp.easing = "EASE_IN_OUT"
# gözler: görsel pikselinden 3B'ye (aynı yüzeyin biraz önüne) parlayan elips
if A.eyes:
    xs = [float(x) for x in A.eyes.split(",")]
    em = bpy.data.materials.new("eye"); em.use_nodes = True; nt = em.node_tree; nt.nodes.clear(); e = nt.nodes.new("ShaderNodeEmission")
    e.inputs[0].default_value = (0.35, 0.8, 1.0, 1); e.inputs[1].default_value = 6.0; nt.links.new(e.outputs[0], nt.nodes.new("ShaderNodeOutputMaterial").inputs[0])
    for k in range(0, len(xs), 2):
        px, py = xs[k], xs[k + 1]; u, v = px / IW, py / IH; z = float(Z[int(py), int(px)]) * 0.985
        bpy.ops.mesh.primitive_uv_sphere_add(radius=1, location=((u * 2 - 1) * tanx * z, z, (1 - v * 2) * tany * z)); s_ = bpy.context.object
        r = 0.018 * z; s_.scale = (r * 1.7, r * 0.3, r * 0.45); s_.data.materials.append(em); base = tuple(s_.scale)
        s_.keyframe_insert("scale", frame=1)
        for bt in [float(x) for x in A.blink.split(",") if x]:
            f = 1 + int(bt * (N - 1)); s_.scale = base; s_.keyframe_insert("scale", frame=f); s_.scale = (base[0], base[1], base[2] * 0.08); s_.keyframe_insert("scale", frame=f + 3); s_.scale = base; s_.keyframe_insert("scale", frame=f + 6)
    st = e.inputs[1]; st.keyframe_insert("default_value", frame=1)
    for ft in [float(x) for x in A.flare.split(",") if x]:
        f = 1 + int(ft * (N - 1)); st.default_value = 6; st.keyframe_insert("default_value", frame=f); st.default_value = 22; st.keyframe_insert("default_value", frame=f + 5); st.default_value = 9; st.keyframe_insert("default_value", frame=f + 25)
# render: emisyon → az örnek yeter; hareket bulanıklığı = gerçek kamera hissi
S.render.engine = "CYCLES"; S.cycles.samples = 6; S.cycles.use_denoising = False; S.render.use_motion_blur = True; S.render.motion_blur_shutter = 0.5
S.view_settings.view_transform = "Standard"; S.render.resolution_x, S.render.resolution_y = A.w, A.h
S.cycles.max_bounces = 0; S.render.film_transparent = False
if os.environ.get("GPU") == "1":
    pr = bpy.context.preferences.addons["cycles"].preferences; pr.compute_device_type = "CUDA"; pr.get_devices()
    for d in pr.devices: d.use = True
    S.cycles.device = "GPU"
w = bpy.data.worlds.new("w"); S.world = w
if A.still >= 0:
    S.frame_set(1 + int(A.still * (N - 1))); S.render.filepath = os.path.abspath(A.out); bpy.ops.render.render(write_still=True); sys.exit(0)
tmp = os.path.abspath(A.out) + "_fr"; shutil.rmtree(tmp, ignore_errors=True); os.makedirs(tmp)
S.render.filepath = tmp + "/f_"; S.render.image_settings.file_format = "PNG"; bpy.ops.render.render(animation=True)
FFX = os.environ.get("FFMPEG", "ffmpeg"); subprocess.run([FFX, "-y", "-v", "error", "-framerate", str(A.fps), "-i", tmp + "/f_%04d.png", "-c:v", "libx264", "-crf", "16", "-preset", "veryfast", "-pix_fmt", "yuv420p", A.out], check=True)
shutil.rmtree(tmp); print("ok", A.out)
