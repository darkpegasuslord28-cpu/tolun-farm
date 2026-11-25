"""TOLUN'UN EVİ — kalıcı 3B set + çekim kütüphanesi (Blender/bpy, tamamen kodla; 25 Kas 2026).
Aynı ev, aynı TOLUN, her videoda farklı çekim/etkileşim. Set: çalışma odası (deniz manzaralı pencere, masa, 2 monitör,
lamba, kitaplık, kedi) + koridor. TOLUN: kapüşonlu, yüzsüz, iki mavi göz (göz kırpma/kısma/parlama anahtar karelerle).
Çekimler (SCENE): pov_enter · pov_look · pov_book · eyes · orbit · corridor · behind · window
    SCENE=pov_enter python3 tolun_ev.py still out.png          # tek kare (yerel kontrol)
    SCENE=orbit python3 tolun_ev.py anim out_dir 30 6           # kareler (Kaggle GPU: GPU=1)
Ortam: W,H, SAMPLES, RAIN=1 (camda yağmur), STORM=1 (şimşek), SCREEN=<png url/yol> (monitör yazısı), SEED. Kişisel bilgi/anahtar YOK."""
import bpy, math, sys, os, random
from mathutils import Vector
args = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
MODE = args[0] if args else "still"; OUT = args[1] if len(args) > 1 else "still.png"
FPS = int(args[2]) if len(args) > 2 else 30; SEC = float(args[3]) if len(args) > 3 else 4
SCENE = os.environ.get("SCENE", "orbit"); RES = (int(os.environ.get("W", 540)), int(os.environ.get("H", 960)))
SAMPLES = int(os.environ.get("SAMPLES", 32)); STORM = os.environ.get("STORM", "1") == "1"; SEED = int(os.environ.get("SEED", 7))
N = int(round(FPS * SEC)); rnd = random.Random(SEED)
bpy.ops.wm.read_factory_settings(use_empty=True); S = bpy.context.scene; S.frame_start, S.frame_end = 1, N; S.render.fps = FPS
F = lambda t: 1 + int(round(t * (N - 1)))          # t: 0..1 → kare

# ---------------- malzemeler ----------------
def mat(name, col, rough=0.6, metal=0.0, emit=None, es=0.0):
    m = bpy.data.materials.new(name); m.use_nodes = True; b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*col, 1); b.inputs["Roughness"].default_value = rough; b.inputs["Metallic"].default_value = metal
    if emit: b.inputs["Emission Color"].default_value = (*emit, 1); b.inputs["Emission Strength"].default_value = es
    return m
def emit_mat(name, col, s):
    m = bpy.data.materials.new(name); m.use_nodes = True; nt = m.node_tree; nt.nodes.clear()
    e = nt.nodes.new("ShaderNodeEmission"); e.inputs["Color"].default_value = (*col, 1); e.inputs["Strength"].default_value = s
    nt.links.new(e.outputs[0], nt.nodes.new("ShaderNodeOutputMaterial").inputs["Surface"]); return m, e
def img_mat(name, path, s=3.0):
    m = bpy.data.materials.new(name); m.use_nodes = True; nt = m.node_tree; nt.nodes.clear()
    t = nt.nodes.new("ShaderNodeTexImage"); t.image = bpy.data.images.load(path); e = nt.nodes.new("ShaderNodeEmission"); e.inputs["Strength"].default_value = s
    nt.links.new(t.outputs[0], e.inputs[0]); nt.links.new(e.outputs[0], nt.nodes.new("ShaderNodeOutputMaterial").inputs["Surface"]); return m
WALL = mat("wall", (0.035, 0.04, 0.055), 0.85); FLOOR = mat("floor", (0.09, 0.06, 0.04), 0.45); WOOD = mat("wood", (0.12, 0.07, 0.035), 0.5)
HOOD = mat("hood", (0.03, 0.03, 0.034), 0.92); HOOD.node_tree.nodes["Principled BSDF"].inputs["Sheen Weight"].default_value = 0.6; VOID = mat("void", (0, 0, 0), 1.0); METAL = mat("metal", (0.3, 0.3, 0.32), 0.35, 0.9)
EYE, EYE_N = emit_mat("eye", (0.25, 0.75, 1.0), 40.0); SCR, SCR_N = emit_mat("screen", (0.15, 0.45, 0.9), 2.2)
LAMPM = emit_mat("lampbulb", (1.0, 0.62, 0.3), 25.0)[0]; MOON = emit_mat("moon", (0.8, 0.85, 1.0), 6.0)[0]
SEA = mat("sea", (0.004, 0.01, 0.02), 0.08, 0.0); SKY, SKY_N = emit_mat("sky", (0.02, 0.035, 0.07), 1.0)
CAT = mat("cat", (0.015, 0.015, 0.018), 0.9)
BOOKC = [mat(f"book{i}", c, 0.7) for i, c in enumerate([(0.25, 0.04, 0.03), (0.05, 0.12, 0.2), (0.18, 0.14, 0.05), (0.06, 0.15, 0.08), (0.2, 0.18, 0.15), (0.1, 0.04, 0.12)])]

def box(name, loc, size, m, rot=(0, 0, 0)):
    bpy.ops.mesh.primitive_cube_add(location=loc, rotation=rot); o = bpy.context.object; o.name = name; o.scale = (size[0] / 2, size[1] / 2, size[2] / 2)
    o.data.materials.append(m); return o
def sph(name, loc, r, m, scale=(1, 1, 1), seg=48):
    bpy.ops.mesh.primitive_uv_sphere_add(location=loc, radius=r, segments=seg, ring_count=seg // 2); o = bpy.context.object; o.name = name; o.scale = scale
    o.data.materials.append(m); bpy.ops.object.shade_smooth(); return o
def cyl(name, loc, r, d, m, rot=(0, 0, 0), v=48):
    bpy.ops.mesh.primitive_cylinder_add(location=loc, radius=r, depth=d, rotation=rot, vertices=v); o = bpy.context.object; o.name = name
    o.data.materials.append(m); bpy.ops.object.shade_smooth(); return o
def light(kind, loc, energy, col, size=0.3, rot=(0, 0, 0)):
    d = bpy.data.lights.new(kind + str(loc), kind); d.energy = energy; d.color = col
    if kind == "AREA": d.size = size
    elif kind in ("POINT", "SPOT"): d.shadow_soft_size = size
    o = bpy.data.objects.new(d.name, d); o.location = loc; o.rotation_euler = rot; S.collection.objects.link(o); return o

# ---------------- ÇALIŞMA ODASI (x: -3..3, y: -4..3, z: 0..3) ----------------
box("floor", (0, -0.5, -0.05), (6.4, 7.4, 0.1), FLOOR); box("ceil", (0, -0.5, 3.05), (6.4, 7.4, 0.1), WALL)
box("wallL", (-3.2, -0.5, 1.5), (0.1, 7.4, 3), WALL); box("wallR", (3.2, -0.5, 1.5), (0.1, 7.4, 3), WALL)
box("wallF_l", (-2.35, -4.2, 1.5), (1.7, 0.1, 3), WALL); box("wallF_r", (1.85, -4.2, 1.5), (2.7, 0.1, 3), WALL); box("wallF_t", (-0.75, -4.2, 2.6), (1.5, 0.1, 0.8), WALL)  # kapı boşluğu x≈-1.5..0
# arka duvar: büyük pencere (x -2.2..2.2, z 0.8..2.7)
box("wallB_b", (0, 3.0, 0.4), (6.4, 0.1, 0.8), WALL); box("wallB_t", (0, 3.0, 2.85), (6.4, 0.1, 0.3), WALL)
box("wallB_l", (-2.7, 3.0, 1.75), (1.0, 0.1, 1.9), WALL); box("wallB_r", (2.7, 3.0, 1.75), (1.0, 0.1, 1.9), WALL)
for x in (-1.1, 0, 1.1): box(f"mull{x}", (x, 3.0, 1.75), (0.05, 0.12, 1.9), WOOD)
box("mullh", (0, 3.0, 1.75), (4.4, 0.12, 0.05), WOOD)
# dışarısı: deniz + gök + ay + uzak burun
box("sea", (0, 40, -0.6), (200, 70, 0.1), SEA); bpy.ops.mesh.primitive_plane_add(location=(0, 70, 20), rotation=(math.pi / 2, 0, 0), size=200)
sky = bpy.context.object; sky.data.materials.append(SKY); sph("moon", (9, 60, 14), 2.2, MOON)
cape = box("cape", (-14, 45, 1.0), (16, 6, 4), mat("cape", (0.005, 0.006, 0.01), 0.9), rot=(0, 0.2, 0.4))
light("SUN", (0, 0, 10), 1.2, (0.55, 0.65, 1.0), rot=(math.radians(-60), 0, math.radians(170)))   # ay ışığı pencereden
# masa (pencereye bakan değil: kameraya bakan; TOLUN pencerenin önünde, yüzü kapıya)
box("desk", (0, 1.4, 0.74), (2.2, 0.9, 0.06), WOOD)
for dx in (-1.0, 1.0):
    for dy in (1.0, 1.8): box(f"leg{dx}{dy}", (dx, dy, 0.36), (0.06, 0.06, 0.72), METAL)
for i, dx in enumerate((-0.72, 0.72)):   # monitörler kameraya sırtı dönük değil: TOLUN'a dönük (yüzü +y), arkası kameraya → ekran ışığı TOLUN'a vurur
    mon = box(f"mon{i}", (dx, 1.05, 1.0), (0.62, 0.04, 0.38), mat(f"monb{i}", (0.01, 0.01, 0.012), 0.4), rot=(0, 0, (-1) ** i * -0.45))
    scr = box(f"scr{i}", (dx - (-1) ** i * 0.01, 1.075, 1.0), (0.58, 0.005, 0.34), SCR, rot=(0, 0, (-1) ** i * -0.45)); box(f"stand{i}", (dx, 1.05, 0.83), (0.05, 0.05, 0.2), METAL)
light("AREA", (0, 1.15, 1.12), 40, (0.3, 0.6, 1.0), size=1.2, rot=(math.radians(-90), 0, 0))     # monitör ışığı TOLUN'a
SCRL = S.objects[-1]
light("AREA", (0, -0.5, 2.9), 60, (0.35, 0.45, 0.8), size=4.0)   # tavandan soğuk dolgu
light("AREA", (-2.4, -0.8, 2.2), 25, (1.0, 0.7, 0.45), size=1.0, rot=(0, math.radians(-60), 0))   # kitaplığa sıcak
# lamba
cyl("lampbase", (-0.85, 1.6, 0.79), 0.09, 0.03, METAL); cyl("lamparm", (-0.85, 1.6, 1.05), 0.012, 0.5, METAL)
sph("bulb", (-0.75, 1.5, 1.3), 0.04, LAMPM); LAMP = light("POINT", (-0.75, 1.45, 1.28), 90, (1.0, 0.6, 0.3), size=0.1)
# kupa
cyl("mug", (0.85, 1.25, 0.82), 0.045, 0.1, mat("mugm", (0.6, 0.58, 0.55), 0.3))
# kitaplık (sol duvar) + kitaplar
for z in (0.3, 0.9, 1.5, 2.1): box(f"shelf{z}", (-2.95, -0.8, z), (0.4, 2.4, 0.04), WOOD)
box("shelfS1", (-2.95, -2.0, 1.2), (0.4, 0.04, 2.0), WOOD); box("shelfS2", (-2.95, 0.4, 1.2), (0.4, 0.04, 2.0), WOOD)
BOOKS = []
for z in (0.3, 0.9, 1.5, 2.1):
    y = -1.95
    while y < 0.3:
        h = rnd.uniform(0.22, 0.34); t = rnd.uniform(0.035, 0.07); tilt = rnd.uniform(-0.05, 0.05) if rnd.random() < 0.2 else 0
        BOOKS.append(box("book", (-2.95, y + t / 2, z + 0.02 + h / 2), (0.24, t, h), rnd.choice(BOOKC), rot=(tilt, 0, 0))); y += t + 0.004
# kedi (masada uyuyor)
catb = sph("catbody", (0.55, 1.7, 0.86), 0.16, CAT, scale=(1.4, 0.9, 0.6)); cath = sph("cathead", (0.33, 1.68, 0.9), 0.09, CAT)
for s in (-1, 1): bpy.ops.mesh.primitive_cone_add(radius1=0.035, depth=0.07, location=(0.31, 1.68 + s * 0.05, 0.98)); bpy.context.object.data.materials.append(CAT)
# halı + sandalye
box("rug", (0, 0.2, 0.005), (2.6, 1.8, 0.01), mat("rug", (0.08, 0.03, 0.03), 0.95))
box("chairseat", (0, 2.0, 0.48), (0.55, 0.5, 0.06), mat("chair", (0.02, 0.02, 0.022), 0.6)); box("chairback", (0, 2.27, 0.85), (0.55, 0.06, 0.7), mat("chair2", (0.02, 0.02, 0.022), 0.6))

# ---------------- TOLUN ----------------
def tolun(loc, rotz=0.0, standing=False):
    """Yüz (kapüşon açıklığı) yerel -y yönünde. standing: ayakta (bacaklar + uzun gövde)."""
    root = bpy.data.objects.new("TOLUN", None); S.collection.objects.link(root); root.location = loc; root.rotation_euler = (0, 0, rotz)
    parts = []; z0 = 0.62 if not standing else 1.05
    if standing:
        for s in (-1, 1): parts.append(cyl(f"leg{s}", (s * 0.1, 0, 0.45), 0.075, 0.9, mat("pants", (0.012, 0.012, 0.014), 0.8)))
    body = cyl("torso", (0, 0, z0), 0.25, 0.8, HOOD); body.scale = (1.0, 0.68, 1.0); parts.append(body)
    sh = sph("shoulders", (0, 0, z0 + 0.36), 0.31, HOOD, scale=(1.12, 0.66, 0.42)); parts.append(sh)
    for s in (-1, 1):   # kapüşon ipleri
        parts.append(cyl(f"cord{s}", (s * 0.06, -0.19, z0 + 0.2), 0.006, 0.3, mat("cord", (0.5, 0.52, 0.55), 0.5)))
    head = bpy.data.objects.new("HEAD", None); S.collection.objects.link(head); head.parent = root; head.location = (0, 0, z0 + 0.62)
    hood = sph("hood", (0, 0.02, 0), 0.2, HOOD, scale=(1.0, 1.12, 1.22)); hood.parent = head
    peak = sph("hoodpeak", (0, 0.06, 0.12), 0.15, HOOD, scale=(0.9, 1.1, 1.05)); peak.parent = head
    bpy.ops.mesh.primitive_torus_add(location=(0, -0.17, -0.01), rotation=(math.pi / 2, 0, 0), major_radius=0.13, minor_radius=0.035)
    rim = bpy.context.object; rim.scale = (0.95, 1.25, 1.0); rim.data.materials.append(HOOD); bpy.ops.object.shade_smooth(); rim.parent = head
    void = sph("void", (0, -0.13, -0.01), 0.14, VOID, scale=(0.92, 0.55, 1.18)); void.parent = head
    eyes = []
    for s in (-1, 1):
        e = sph(f"eye{s}", (s * 0.05, -0.205, 0.015), 0.022, EYE, scale=(1.6, 0.4, 0.35), seg=24); e.parent = head; e.rotation_euler = (0, s * 0.18, 0); eyes.append(e)
    for s in (-1, 1):
        arm = cyl(f"arm{s}", (s * 0.31, -0.12, z0 + 0.08), 0.075, 0.45, HOOD, rot=(math.radians(55), 0, 0)); parts.append(arm)
        cuff = cyl(f"cuff{s}", (s * 0.3, -0.31, z0 - 0.04), 0.068, 0.08, VOID, rot=(math.radians(55), 0, 0)); parts.append(cuff)
    for p_ in parts + [hood, peak, rim]:
        if p_.data.materials and p_.data.materials[0] == HOOD:
            bpy.context.view_layer.objects.active = p_; m1 = p_.modifiers.new("sub", "SUBSURF"); m1.levels = m1.render_levels = 2
            tx = bpy.data.textures.new("fold" + p_.name, "CLOUDS"); tx.noise_scale = 0.09; d = p_.modifiers.new("fold", "DISPLACE"); d.texture = tx; d.strength = 0.018
    for p_ in parts: p_.parent = root
    eyel = light("POINT", (0, -0.32, z0 + 0.63), 1.2, (0.3, 0.75, 1.0), size=0.05); eyel.parent = root
    return root, head, eyes
STAND = SCENE in ("behind", "corridor")
T_ROOT, T_HEAD, T_EYES = tolun((0, 1.95, 0.0) if not STAND else (0, 0, 0), 0.0, STAND)   # sandalyede, yüzü kapıya (-y)

# ---------------- KORİDOR (kapının dışı: y -4.2 .. -14) ----------------
CW = mat("cwall", (0.03, 0.032, 0.04), 0.9)
box("cfloor", (-0.75, -9.2, -0.05), (1.6, 10, 0.1), FLOOR); box("cceil", (-0.75, -9.2, 2.6), (1.6, 10, 0.1), CW)
box("cwl", (-1.6, -9.2, 1.3), (0.1, 10, 2.6), CW); box("cwr", (0.1, -9.2, 1.3), (0.1, 10, 2.6), CW)
box("cend", (-0.75, -14.2, 1.3), (1.6, 0.1, 2.6), emit_mat("cendl", (0.5, 0.7, 1.0), 2.5)[0])
CLIGHTS = []
for k, y in enumerate((-6, -9, -12)):
    box(f"cl{k}", (-0.75, y, 2.54), (0.5, 0.15, 0.02), emit_mat(f"cle{k}", (0.8, 0.88, 1.0), 6)[0]); CLIGHTS.append(light("AREA", (-0.75, y, 2.45), 25, (0.75, 0.85, 1.0), size=0.5, rot=(0, 0, 0)))

# ---------------- dünya, render ----------------
w = bpy.data.worlds.new("w"); w.use_nodes = True; w.node_tree.nodes["Background"].inputs[0].default_value = (0.002, 0.003, 0.006, 1); S.world = w
S.render.engine = "CYCLES"; S.cycles.samples = SAMPLES; S.cycles.use_denoising = True
S.render.resolution_x, S.render.resolution_y = RES; S.view_settings.view_transform = "AgX"; S.view_settings.exposure = 0.8; S.view_settings.look = "AgX - Punchy" if "AgX - Punchy" in [l for l in ("AgX - Punchy",)] else "None"
try: S.view_settings.look = "AgX - Punchy"
except Exception: pass
if os.environ.get("GPU") == "1":
    p = bpy.context.preferences.addons["cycles"].preferences; p.compute_device_type = "CUDA"; p.get_devices()
    for d in p.devices: d.use = True
    S.cycles.device = "GPU"

# ---------------- canlı ayrıntılar (her çekimde) ----------------
def key(o, path, f, v):
    setattr(o, path, v) if not isinstance(v, tuple) else setattr(o, path, v); o.keyframe_insert(path, frame=f)
def eyes_anim(events):
    """events: [(t0..1, 'blink'|'squint'|'flare'|'open')]. Göz şekli = ölçek; parlama = Emission gücü."""
    base = (1.6, 0.4, 0.35)
    for e in T_EYES: e.scale = base; e.keyframe_insert("scale", frame=1)
    EYE_N.inputs["Strength"].default_value = 40; EYE_N.inputs["Strength"].keyframe_insert("default_value", frame=1)
    for t, kind in events:
        f = F(t)
        for e in T_EYES:
            if kind == "blink":
                e.scale = base; e.keyframe_insert("scale", frame=f); e.scale = (1.6, 0.4, 0.04); e.keyframe_insert("scale", frame=f + 3); e.scale = base; e.keyframe_insert("scale", frame=f + 6)
            elif kind == "squint":
                e.scale = base; e.keyframe_insert("scale", frame=f); e.scale = (1.7, 0.4, 0.15); e.keyframe_insert("scale", frame=f + 6)
            elif kind == "open":
                e.scale = (1.7, 0.4, 0.15); e.keyframe_insert("scale", frame=f); e.scale = base; e.keyframe_insert("scale", frame=f + 6)
        if kind == "flare":
            n = EYE_N.inputs["Strength"]; n.default_value = 40; n.keyframe_insert("default_value", frame=f); n.default_value = 140; n.keyframe_insert("default_value", frame=f + 5); n.default_value = 60; n.keyframe_insert("default_value", frame=f + 25)
def head_anim(keys):   # keys: [(t, yaw_derece, pitch_derece)]
    for t, yaw, pitch in keys: T_HEAD.rotation_euler = (math.radians(pitch), 0, math.radians(yaw)); T_HEAD.keyframe_insert("rotation_euler", frame=F(t))
def breathe():
    for k in range(0, N + 1, 45):
        T_ROOT.scale = (1, 1, 1.0 + (0.008 if (k // 45) % 2 else 0)); T_ROOT.keyframe_insert("scale", frame=k + 1)
def storm():
    if not STORM: return
    s = SKY_N.inputs["Strength"]; s.default_value = 1.0; s.keyframe_insert("default_value", frame=1)
    for t in (rnd.uniform(0.25, 0.4), rnd.uniform(0.7, 0.85)):
        f = F(t)
        for df, v in ((0, 1.0), (1, 25), (2, 3), (4, 30), (8, 1.0)): s.default_value = v; s.keyframe_insert("default_value", frame=f + df)
def flicker(objs, p=0.15):
    for o in objs:
        e0 = o.data.energy
        for f in range(1, N + 1, 3):
            o.data.energy = e0 * (0.05 if rnd.random() < p else 1.0); o.data.keyframe_insert("energy", frame=f)
def cat_tail():
    catb.rotation_euler = (0, 0, 0); catb.keyframe_insert("rotation_euler", frame=1)
    for k in range(1, N + 1, 40): catb.scale = (1.4, 0.9, 0.6 + (0.03 if (k // 40) % 2 else 0)); catb.keyframe_insert("scale", frame=k)

# ---------------- kamera ----------------
cam_d = bpy.data.cameras.new("cam"); CAM = bpy.data.objects.new("CAM", cam_d); S.collection.objects.link(CAM); S.camera = CAM
TGT = bpy.data.objects.new("TGT", None); S.collection.objects.link(TGT)
c = CAM.constraints.new("TRACK_TO"); c.target = TGT; c.track_axis = "TRACK_NEGATIVE_Z"; c.up_axis = "UP_Y"
cam_d.dof.use_dof = True; cam_d.dof.focus_object = TGT; cam_d.dof.aperture_fstop = 2.2
def path(cam_keys, tgt_keys, lens=24):
    cam_d.lens = lens
    for t, p in cam_keys: CAM.location = p; CAM.keyframe_insert("location", frame=F(t))
    for t, p in tgt_keys: TGT.location = p; TGT.keyframe_insert("location", frame=F(t))
def bob(amp=0.025, freq=1.8):   # birinci şahıs yürüyüş sallantısı (gürültü modifiyeri)
    CAM.animation_data_create()
    for fc in CAM.animation_data.action.fcurves if CAM.animation_data.action and hasattr(CAM.animation_data.action, "fcurves") else []:
        if fc.data_path == "location" and fc.array_index == 2:
            m = fc.modifiers.new("NOISE"); m.scale = FPS / freq; m.strength = amp * 2
EYE_HEIGHT = 1.6
T_HEAD_POS = Vector((0, 1.78, 1.25))

# ---------------- ÇEKİM KÜTÜPHANESİ ----------------
breathe(); storm(); cat_tail()
if SCENE == "pov_enter":        # koridordan odaya gir: kapı → sola pencereye bak → sağa TOLUN'a dön; o da başını çevirir, gözler parlar
    path([(0, (-0.75, -6.5, EYE_HEIGHT)), (0.55, (-0.75, -3.2, EYE_HEIGHT)), (1, (-0.4, -2.0, EYE_HEIGHT))],
         [(0, (-0.75, -3.0, 1.5)), (0.45, (-0.2, 3.0, 1.6)), (0.7, (1.2, 3.0, 1.4)), (1, (0, 1.85, 1.25))], lens=22)
    head_anim([(0, 35, 8), (0.72, 35, 8), (0.85, 0, 0), (1, 0, -3)]); eyes_anim([(0.3, "blink"), (0.88, "flare")]); bob()
elif SCENE == "pov_look":       # odada dur, çevreyi incele: kitaplık → masa → pencere → TOLUN
    path([(0, (0.6, -1.8, EYE_HEIGHT)), (1, (0.4, -1.2, EYE_HEIGHT))],
         [(0, (-2.9, -0.8, 1.5)), (0.3, (-2.9, 0.0, 0.9)), (0.55, (0.8, 1.25, 0.85)), (0.75, (0.5, 3.0, 1.8)), (1, (0, 1.85, 1.25))], lens=26)
    eyes_anim([(0.2, "blink"), (0.82, "squint"), (0.95, "open")]); head_anim([(0, -10, 5), (0.8, -10, 5), (0.95, 15, 0), (1, 15, 0)])
elif SCENE == "pov_book":       # kitaplığa yürü, bir kitap rafından çıkıp kameraya gelir ve açılır
    b = sorted(BOOKS, key=lambda o: abs(o.location.z - 1.5) + abs(o.location.y + 0.8))[0]; p0 = b.location.copy()
    path([(0, (-0.5, -1.5, EYE_HEIGHT)), (0.4, (-2.1, -0.9, 1.55)), (1, (-2.0, -0.9, 1.5))], [(0, (-2.9, -0.8, 1.6)), (0.4, p0), (1, (-2.35, -0.9, 1.38))], lens=28)
    b.location = p0; b.keyframe_insert("location", frame=F(0.45)); b.rotation_euler = (0, 0, 0); b.keyframe_insert("rotation_euler", frame=F(0.45))
    b.location = (p0.x + 0.25, p0.y, p0.z + 0.02); b.keyframe_insert("location", frame=F(0.6))
    b.location = (-2.45, -0.9, 1.36); b.rotation_euler = (math.radians(-20), 0, math.radians(-90)); b.keyframe_insert("location", frame=F(0.85)); b.keyframe_insert("rotation_euler", frame=F(0.85))
    b.scale = (b.scale.x * 1.0, b.scale.y * 4.0, b.scale.z); b.keyframe_insert("scale", frame=F(1.0))   # açılma hissi
elif SCENE == "eyes":           # gözlere makro: kırp → kıs → parla
    path([(0, (0.05, 1.15, 1.27)), (1, (0.03, 1.38, 1.26))], [(0, tuple(T_HEAD_POS)), (1, tuple(T_HEAD_POS))], lens=70)
    cam_d.dof.aperture_fstop = 1.4; eyes_anim([(0.15, "blink"), (0.4, "squint"), (0.65, "open"), (0.72, "flare")]); head_anim([(0, 0, 0), (0.4, 8, 3), (0.7, -4, 0), (1, 0, 0)])
elif SCENE == "orbit":          # masadaki TOLUN etrafında yavaş yay, şimşek pencerede
    keys = [(i / 8, (math.sin(-0.9 + 1.8 * i / 8) * 2.4, 1.95 - math.cos(-0.9 + 1.8 * i / 8) * 2.4, 1.35 + 0.15 * i / 8)) for i in range(9)]
    path(keys, [(0, (0, 1.9, 1.15)), (1, (0, 1.9, 1.22))], lens=35); eyes_anim([(0.3, "blink"), (0.6, "squint"), (0.85, "open")]); head_anim([(0, -20, 0), (0.5, 0, 5), (1, 20, 0)])
elif SCENE == "corridor":       # koridorda yürü, ışıklar titrer, sonda TOLUN silueti
    T_ROOT.location = (-0.75, -13.4, 0); T_ROOT.rotation_euler = (0, 0, math.pi)
    path([(0, (-0.75, -4.6, EYE_HEIGHT)), (1, (-0.75, -10.5, EYE_HEIGHT))], [(0, (-0.75, -14, 1.4)), (1, (-0.75, -14, 1.5))], lens=24); flicker(CLIGHTS, 0.18); bob()
    eyes_anim([(0.5, "blink"), (0.9, "flare")])
elif SCENE == "behind":         # pencereden denize bakarken dön: TOLUN arkanda duruyor
    T_ROOT.location = (0.2, 0.9, 0); T_ROOT.rotation_euler = (0, 0, math.pi)
    path([(0, (0.0, 2.6, EYE_HEIGHT)), (1, (0.05, 2.5, EYE_HEIGHT))], [(0, (0, 30, 1.0)), (0.55, (0, 30, 1.0)), (0.75, (0.2, 0.9, 1.6)), (1, (0.2, 0.9, 1.62))], lens=24)
    eyes_anim([(0.78, "flare")]); head_anim([(0, 0, -6), (1, 0, -6)])
elif SCENE == "window":         # pencere dışı geniş: deniz, ay, şimşek; kamera yavaşça cama yaklaşır
    path([(0, (0.3, 1.0, 1.7)), (1, (0.2, 2.6, 1.75))], [(0, (2, 40, 3)), (1, (3, 40, 4))], lens=30)
else:
    raise SystemExit("bilinmeyen SCENE " + SCENE)

for ob in bpy.data.objects:   # anahtar kareler: yumuşak geçiş
    ad = ob.animation_data
    if ad and ad.action and hasattr(ad.action, "fcurves"):
        for fc in ad.action.fcurves:
            for kp in fc.keyframe_points: kp.interpolation = "BEZIER"

if MODE == "still":
    S.frame_set(int(os.environ.get("FRAME", max(1, N // 2)))); S.render.filepath = OUT; bpy.ops.render.render(write_still=True)
else:
    os.makedirs(OUT, exist_ok=True); S.render.filepath = os.path.join(OUT, "f_"); S.render.image_settings.file_format = "PNG"; bpy.ops.render.render(animation=True)
