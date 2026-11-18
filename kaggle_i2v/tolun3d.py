"""TOLUN 3B SAHNELER (Blender/bpy, tamamen kodla) — HER SAHNEDE AYNI TOLUN ŞİŞESİ (Oğulcan 18 Kas: "aynı şişe üzerinden, diğer videolardaki mantıkla").
Nuit Dorée hikâyesi: candle (mum yanar → söner) · reveal (şişe karanlıktan doğar, ışık camda süzülür) · desert (gün batımı kumul) · city (gece şehir bokeh) · marble (mermer kaide final).
    SCENE=marble python3 tolun3d.py still out.png        # tek kare (yerel kontrol)
    SCENE=marble python3 tolun3d.py anim out_dir 30 6     # kareler (Kaggle GPU: GPU=1)
Ortam: W,H (çözünürlük), SAMPLES. bpy 4.x/5.x uyumlu. Kişisel bilgi/anahtar YOK (tolun-farm public)."""
import bpy, math, sys, os, random
args = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
MODE = args[0] if args else "still"; OUT = args[1] if len(args) > 1 else "still.png"
FPS = int(args[2]) if len(args) > 2 else 30; SEC = float(args[3]) if len(args) > 3 else 3
SCENE = os.environ.get("SCENE", "reveal"); RES = (int(os.environ.get("W", 540)), int(os.environ.get("H", 960))); SAMPLES = int(os.environ.get("SAMPLES", 32))
N = int(round(FPS * SEC))
bpy.ops.wm.read_factory_settings(use_empty=True); S = bpy.context.scene; S.frame_start, S.frame_end = 1, N; S.render.fps = FPS
rnd = random.Random(7)


def mat(name, **k):
    m = bpy.data.materials.new(name); m.use_nodes = True; b = m.node_tree.nodes["Principled BSDF"]
    for key, v in k.items(): b.inputs[key].default_value = v
    return m


def emit_mat(name, col, strength):
    m = bpy.data.materials.new(name); m.use_nodes = True; nt = m.node_tree; nt.nodes.clear()
    e = nt.nodes.new("ShaderNodeEmission"); e.inputs["Color"].default_value = (*col, 1); e.inputs["Strength"].default_value = strength
    nt.links.new(e.outputs[0], nt.nodes.new("ShaderNodeOutputMaterial").inputs["Surface"]); return m


GOLD = mat("gold", **{"Base Color": (1.0, 0.72, 0.33, 1), "Metallic": 1.0, "Roughness": 0.18})


def bottle(x=0.0, y=0.0, z=0.0):
    """TEK ÜRÜN: kalın kristal gövde + amber sıvı (hacim emilimi) + altın boyun/bilezik + mat siyah kapak + altın TOLUN yazısı."""
    GLASS = mat("glass", **{"Base Color": (1, 1, 1, 1), "Roughness": 0.02, "IOR": 1.5, "Transmission Weight": 1.0})
    LIQ = mat("liquid", **{"Base Color": (1, 1, 1, 1), "Roughness": 0.0, "IOR": 1.36, "Transmission Weight": 1.0})
    va = LIQ.node_tree.nodes.new("ShaderNodeVolumeAbsorption"); va.inputs["Color"].default_value = (1.0, 0.42, 0.06, 1); va.inputs["Density"].default_value = 3.0
    LIQ.node_tree.links.new(va.outputs[0], LIQ.node_tree.nodes["Material Output"].inputs["Volume"])
    BLACK = mat("cap", **{"Base Color": (0.004, 0.004, 0.005, 1), "Roughness": 0.5})
    def cube(loc, sc, m, bev=0.0, seg=1):
        bpy.ops.mesh.primitive_cube_add(size=1, location=(x + loc[0], y + loc[1], z + loc[2])); o = bpy.context.object; o.scale = sc
        if bev: md = o.modifiers.new("b", "BEVEL"); md.width = bev; md.segments = seg
        o.data.materials.append(m); return o
    body = cube((0, 0, 0.75), (0.9, 0.42, 1.5), GLASS, 0.05, 6); body.modifiers.new("s", "SOLIDIFY").thickness = 0.06
    cube((0, 0, 0.66), (0.76, 0.30, 1.04), LIQ, 0.02)
    bpy.ops.mesh.primitive_cylinder_add(radius=0.11, depth=0.12, location=(x, y, z + 1.56)); bpy.context.object.data.materials.append(GOLD)
    cube((0, 0, 1.63), (0.44, 0.32, 0.025), GOLD); cube((0, 0, 1.79), (0.42, 0.30, 0.34), BLACK, 0.02, 2)
    bpy.ops.object.text_add(location=(x, y - 0.212, z + 0.55)); t = bpy.context.object; t.data.body = "TOLUN"; t.data.align_x = "CENTER"; t.data.size = 0.11
    t.data.extrude = 0.004; t.rotation_euler = (math.radians(90), 0, 0); t.data.space_character = 1.6; t.data.materials.append(GOLD)


def light(name, loc, size, energy, col, rot, kind="AREA", trans=False):
    bpy.ops.object.light_add(type=kind, location=loc, rotation=rot); L = bpy.context.object; L.name = name; L.data.energy = energy; L.data.color = col
    if kind == "AREA": L.data.size = size
    else: L.data.shadow_soft_size = size
    L.visible_camera = False; L.visible_transmission = trans; return L


def world(col=(0, 0, 0), strength=1.0):
    W = bpy.data.worlds.new("w"); S.world = W; W.use_nodes = True; b = W.node_tree.nodes["Background"]; b.inputs[0].default_value = (*col, 1); b.inputs[1].default_value = strength; return W


def plane(size, loc, m, rot=(0, 0, 0)):
    bpy.ops.mesh.primitive_plane_add(size=size, location=loc, rotation=rot); o = bpy.context.object; o.data.materials.append(m); return o


def glow_card(loc, size, col, strength, scale=1.1):
    """uzak, yumuşak radyal parıltı (sıvının içinden altın görünsün)"""
    m = bpy.data.materials.new("glow"); m.use_nodes = True; nt = m.node_tree; nt.nodes.clear()
    tc = nt.nodes.new("ShaderNodeTexCoord"); mp = nt.nodes.new("ShaderNodeMapping"); mp.inputs["Scale"].default_value = (scale, scale, 1)
    gr = nt.nodes.new("ShaderNodeTexGradient"); gr.gradient_type = "SPHERICAL"; cr = nt.nodes.new("ShaderNodeValToRGB")
    cr.color_ramp.elements[0].color = (0, 0, 0, 1); cr.color_ramp.elements[1].color = (*col, 1); cr.color_ramp.elements[1].position = 0.9
    em = nt.nodes.new("ShaderNodeEmission"); em.inputs["Strength"].default_value = strength; out = nt.nodes.new("ShaderNodeOutputMaterial")
    for a, b in ((tc.outputs["Object"], mp.inputs["Vector"]), (mp.outputs["Vector"], gr.inputs["Vector"]), (gr.outputs["Fac"], cr.inputs["Fac"]), (cr.outputs["Color"], em.inputs["Color"]), (em.outputs[0], out.inputs["Surface"])):
        nt.links.new(a, b)
    return plane(size, loc, m, (math.radians(90), 0, 0))


def camera(path, target=(0, 0, 0.95), lens=60, fstop=2.8, ease=True):
    """path: [(kare, (x,y,z)), ...] — yavaş, yumuşak kamera hareketi (her kare hareketli)"""
    bpy.ops.object.empty_add(location=target); tgt = bpy.context.object
    bpy.ops.object.camera_add(); cam = bpy.context.object; S.camera = cam; cam.data.lens = lens
    cam.data.dof.use_dof = True; cam.data.dof.focus_object = tgt; cam.data.dof.aperture_fstop = fstop
    c = cam.constraints.new("TRACK_TO"); c.target = tgt; c.track_axis = "TRACK_NEGATIVE_Z"; c.up_axis = "UP_Y"
    for f, loc in path: cam.location = loc; cam.keyframe_insert("location", frame=f)
    return cam, tgt


def key(obj, attr, frames_vals, data=False):
    o = obj.data if data else obj
    for f, v in frames_vals: setattr(o, attr, v); o.keyframe_insert(attr, frame=f)


# ───────────── SAHNELER ─────────────
def sc_candle():
    """karanlık: yalnız mum ışığı şişeyi yalar; alev titrer, son 0,6 sn'de söner (sonraki kesme: duman)"""
    world((0, 0, 0)); bottle(); FL = mat("fl", **{"Base Color": (0.003, 0.003, 0.003, 1), "Roughness": 0.15}); plane(30, (0, 0, 0), FL)
    WAX = mat("wax", **{"Base Color": (0.9, 0.82, 0.68, 1), "Roughness": 0.6, "Subsurface Weight": 0.4})
    bpy.ops.mesh.primitive_cylinder_add(radius=0.16, depth=0.95, location=(0.85, -0.55, 0.475)); bpy.context.object.data.materials.append(WAX)
    bpy.ops.mesh.primitive_cylinder_add(radius=0.008, depth=0.06, location=(0.85, -0.55, 0.98)); bpy.context.object.data.materials.append(mat("wick", **{"Base Color": (0.02, 0.02, 0.02, 1)}))
    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.035, location=(0.85, -0.55, 1.04)); flame = bpy.context.object; flame.scale = (1, 1, 2.4)
    flame.data.materials.append(emit_mat("flame", (1.0, 0.55, 0.18), 30)); flame.visible_shadow = False
    L = light("candle", (0.85, -0.55, 1.09), 0.04, 40, (1.0, 0.55, 0.2), (0, 0, 0), kind="POINT")
    out_f = N - int(0.6 * FPS)
    for f in range(1, N + 1, 2):                                           # titreşim (sabit tohum)
        k = 1.0 if f < out_f else max(0.0, 1 - (f - out_f) / (0.35 * FPS))
        e = (40 + rnd.uniform(-9, 9)) * k; L.data.energy = e; L.data.keyframe_insert("energy", frame=f)
        flame.scale = (k * (1 + rnd.uniform(-.08, .08)), k * (1 + rnd.uniform(-.08, .08)), k * (2.4 + rnd.uniform(-.3, .3))); flame.keyframe_insert("scale", frame=f)
    glow_card((0, 4, 0.9), 7, (1.0, 0.5, 0.15), 0.55)
    for nm, x, rz in (("rimL", -2.2, -130), ("rimR", 2.2, 130)): light(nm, (x, 1.8, 1.4), 1.2, 260, (1.0, 0.6, 0.3), (math.radians(80), 0, math.radians(rz)))
    camera([(1, (-0.9, -5.2, 1.0)), (N, (-0.4, -4.5, 1.05))], target=(0.2, 0, 0.9), lens=50, fstop=2.0)


def sc_reveal():
    """şişe karanlıktan doğar: kontur ışıkları ve arka hale yükselir; ince ışık şeridi camda soldan sağa süzülür; yavaş yörünge"""
    world((0, 0, 0)); bottle(); plane(30, (0, 0, 0), mat("fl", **{"Base Color": (0.004, 0.004, 0.004, 1), "Roughness": 0.08}))
    g = glow_card((0, 4, 0.8), 9, (1.0, 0.55, 0.16), 1.6)
    em = g.data.materials[0].node_tree.nodes["Emission"]; [(em.inputs["Strength"].__setattr__("default_value", v), em.inputs["Strength"].keyframe_insert("default_value", frame=f)) for f, v in ((1, 0.0), (int(N * 0.7), 1.6))]
    for nm, x, rz in (("rimL", -2.2, -130), ("rimR", 2.2, 130)):
        L = light(nm, (x, 1.8, 1.4), 1.2, 900, (1.0, 0.68, 0.35), (math.radians(80), 0, math.radians(rz))); key(L, "energy", ((1, 0), (int(N * 0.6), 900)), data=True)
    light("top", (0, 1.2, 4.0), 2.0, 60, (1.0, 0.9, 0.8), (0, 0, 0))
    sw = light("sweep", (-3, -2.5, 1.0), 0.12, 3, (1.0, 0.85, 0.6), (math.radians(90), 0, 0)); sw.data.shape = "RECTANGLE"; sw.data.size_y = 2.6
    key(sw, "location", ((1, (-3, -2.5, 1.0)), (N, (3, -2.5, 1.0))))
    camera([(1, (6.4 * math.sin(math.radians(-14)), -6.4 * math.cos(math.radians(-14)), 1.35)), (N, (5.8 * math.sin(math.radians(10)), -5.8 * math.cos(math.radians(10)), 1.3))])


def sc_desert():
    """gün batımı kumul: alçak açı, şişe kumda gerçek gölgeyle; kamera yana kayar (paralaks), arkada güneş"""
    W = world(); nt = W.node_tree; bg = nt.nodes["Background"]; bg.inputs[1].default_value = 1.0
    tc = nt.nodes.new("ShaderNodeTexCoord"); sep = nt.nodes.new("ShaderNodeSeparateXYZ"); cr = nt.nodes.new("ShaderNodeValToRGB")
    cr.color_ramp.elements[0].position = 0.0; cr.color_ramp.elements[0].color = (1.0, 0.42, 0.12, 1); cr.color_ramp.elements[1].position = 0.35; cr.color_ramp.elements[1].color = (0.06, 0.05, 0.12, 1)
    nt.links.new(tc.outputs["Generated"], sep.inputs[0]); nt.links.new(sep.outputs["Z"], cr.inputs["Fac"]); nt.links.new(cr.outputs["Color"], bg.inputs["Color"])
    bpy.ops.mesh.primitive_grid_add(x_subdivisions=260, y_subdivisions=260, size=80, location=(0, 22, 0)); dune = bpy.context.object
    tx = bpy.data.textures.new("dn", "CLOUDS"); tx.noise_scale = 5.0; d = dune.modifiers.new("d", "DISPLACE"); d.texture = tx; d.strength = 3.5; d.mid_level = 0.5
    dune.modifiers.new("ss", "SUBSURF").levels = 1
    SAND = mat("sand", **{"Base Color": (0.85, 0.52, 0.28, 1), "Roughness": 0.95}); st = SAND.node_tree; nz = st.nodes.new("ShaderNodeTexNoise"); nz.inputs["Scale"].default_value = 60
    bm = st.nodes.new("ShaderNodeBump"); bm.inputs["Strength"].default_value = 0.25; st.links.new(nz.outputs["Fac"], bm.inputs["Height"]); st.links.new(bm.outputs["Normal"], st.nodes["Principled BSDF"].inputs["Normal"]); dune.data.materials.append(SAND)
    bpy.ops.object.shade_smooth()
    dg = bpy.context.evaluated_depsgraph_get(); me = dune.evaluated_get(dg).to_mesh()
    h = min(me.vertices, key=lambda v: (v.co.x * dune.scale.x + dune.location.x) ** 2 + (v.co.y + dune.location.y) ** 2).co.z; dune.location.z -= h + 0.01   # şişe kuma otursun
    dune.evaluated_get(dg).to_mesh_clear()
    sun = light("sun", (0, 30, 3), 0.0, 5.0, (1.0, 0.6, 0.3), (math.radians(80), 0, math.radians(-140)), kind="SUN"); sun.data.angle = math.radians(1.5)
    bpy.ops.mesh.primitive_uv_sphere_add(radius=2.2, location=(-4, 60, 4.5)); bpy.context.object.data.materials.append(emit_mat("sundisk", (1.0, 0.72, 0.4), 6))
    light("fill", (-2, -3, 1.5), 3, 40, (1.0, 0.7, 0.5), (math.radians(70), 0, math.radians(-30)))
    for nm, x, rz in (("rimL", -2.2, -130), ("rimR", 2.2, 130)): light(nm, (x, 1.8, 1.4), 1.2, 500, (1.0, 0.68, 0.35), (math.radians(80), 0, math.radians(rz)))
    bottle()
    camera([(1, (-1.6, -4.6, 0.45)), (N, (-0.6, -4.2, 0.5))], target=(0, 0, 0.75), lens=45, fstop=2.2)


def sc_city():
    """gece şehir: ıslak siyah çatı kenarı; uzakta yüzlerce ışık bokeh olur; kamera yana kayar (paralaks)"""
    world((0.004, 0.006, 0.014)); bottle()
    plane(30, (0, 0, 0), mat("wet", **{"Base Color": (0.01, 0.01, 0.012, 1), "Roughness": 0.06}))
    cols = [(1.0, 0.62, 0.25), (1.0, 0.8, 0.5), (0.45, 0.6, 1.0), (1.0, 0.35, 0.2)]
    mats = [emit_mat(f"c{i}", c, 14) for i, c in enumerate(cols)]
    for i in range(260):
        bpy.ops.mesh.primitive_uv_sphere_add(radius=rnd.uniform(0.08, 0.2), segments=8, ring_count=6, location=(rnd.uniform(-22, 22), rnd.uniform(26, 50), rnd.uniform(-1.5, 9)))
        bpy.context.object.data.materials.append(mats[rnd.choice((0, 0, 1, 1, 2, 3))])
    for nm, x, rz in (("rimL", -2.2, -130), ("rimR", 2.2, 130)): light(nm, (x, 1.8, 1.4), 1.2, 700, (1.0, 0.68, 0.35), (math.radians(80), 0, math.radians(rz)))
    light("key", (1.5, -3, 2.2), 1.5, 90, (1.0, 0.75, 0.5), (math.radians(60), 0, math.radians(30)))
    light("cool", (-3, -2, 2.5), 2, 120, (0.5, 0.65, 1.0), (math.radians(60), 0, math.radians(-50)))
    camera([(1, (-1.4, -5.6, 1.25)), (N, (1.0, -5.4, 1.3))], lens=70, fstop=1.4)


def sc_marble():
    """final: mermer kaide üstünde şişe, sıcak tepe ışığı; geniş plandan şişeye yaklaşır; alt üçte bir yazıya boş"""
    world((0, 0, 0)); M = mat("marble", **{"Roughness": 0.18}); nt = M.node_tree; b = nt.nodes["Principled BSDF"]
    wv = nt.nodes.new("ShaderNodeTexWave"); wv.inputs["Scale"].default_value = 1.6; wv.inputs["Distortion"].default_value = 9; wv.inputs["Detail"].default_value = 6
    cr = nt.nodes.new("ShaderNodeValToRGB"); cr.color_ramp.elements[0].color = (0.62, 0.58, 0.54, 1); cr.color_ramp.elements[0].position = 0.0
    cr.color_ramp.elements[1].color = (0.92, 0.9, 0.86, 1); cr.color_ramp.elements[1].position = 0.06
    nt.links.new(wv.outputs["Fac"], cr.inputs["Fac"]); nt.links.new(cr.outputs["Color"], b.inputs["Base Color"])
    bpy.ops.mesh.primitive_cylinder_add(radius=1.25, depth=1.2, location=(0, 0, -0.6), vertices=128); ped = bpy.context.object; ped.data.materials.append(M)
    ped.modifiers.new("b", "BEVEL").width = 0.02; bpy.ops.object.shade_smooth()
    plane(40, (0, 0, -1.2), mat("fl", **{"Base Color": (0.004, 0.004, 0.004, 1), "Roughness": 0.7})); bottle()
    glow_card((0, 6, 1.0), 14, (1.0, 0.55, 0.16), 1.2, scale=0.8)
    light("key", (0, -0.6, 4.2), 1.2, 450, (1.0, 0.8, 0.55), (math.radians(8), 0, 0))
    for nm, x, rz in (("rimL", -2.2, -130), ("rimR", 2.2, 130)): light(nm, (x, 1.8, 1.4), 1.2, 700, (1.0, 0.68, 0.35), (math.radians(80), 0, math.radians(rz)))
    camera([(1, (0, -9.5, 0.2)), (N, (0, -5.6, 0.85))], target=(0, 0, 0.55), lens=50, fstop=4.0)


{"candle": sc_candle, "reveal": sc_reveal, "desert": sc_desert, "city": sc_city, "marble": sc_marble}[SCENE]()
for o in bpy.data.objects:
    if o.animation_data and o.animation_data.action:
        for fc in getattr(o.animation_data.action, "fcurves", []):
            for kp in fc.keyframe_points: kp.interpolation = "BEZIER"
S.render.engine = "CYCLES"; S.cycles.samples = SAMPLES; S.cycles.use_denoising = True; S.cycles.max_bounces = 12; S.cycles.transmission_bounces = 12
S.render.resolution_x, S.render.resolution_y = RES; S.view_settings.view_transform = "AgX"
try: S.view_settings.look = "AgX - Punchy"
except Exception: pass
if os.environ.get("GPU"):
    p = bpy.context.preferences.addons["cycles"].preferences; p.compute_device_type = "CUDA"; p.get_devices(); [setattr(d, "use", True) for d in p.devices]; S.cycles.device = "GPU"
if MODE == "still":
    S.frame_set(int(os.environ.get("FRAME", N // 2))); S.render.filepath = OUT; bpy.ops.render.render(write_still=True)
else:
    os.makedirs(OUT, exist_ok=True); S.render.filepath = os.path.join(OUT, "f_"); S.render.image_settings.file_format = "PNG"; bpy.ops.render.render(animation=True)
print("ok", SCENE)
