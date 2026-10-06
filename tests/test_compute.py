# SPDX-License-Identifier: GPL-3.0-or-later
"""Headless tests against shapes with known answers.

    blender -b --factory-startup --python-exit-code 1 --python tests/test_compute.py
"""

import math
import random
import sys
from pathlib import Path

import bmesh
import bpy
from mathutils import Euler, Vector

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import center_of_mass_viz as addon  # noqa: E402
from center_of_mass_viz import compute  # noqa: E402

failures = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}", flush=True)
    if not ok:
        failures.append(name)


def close(a, b, tol=1e-6):
    return (Vector(a) - Vector(b)).length <= tol


def clear_scene():
    for ob in list(bpy.data.objects):
        bpy.data.objects.remove(ob)


def mesh_object(name, verts, faces):
    me = bpy.data.meshes.new(name)
    me.from_pydata(verts, [], faces)
    me.update()
    ob = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(ob)
    return ob


def box(name, lo, hi):
    (x0, y0, z0), (x1, y1, z1) = lo, hi
    verts = [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
             (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]
    faces = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
    return mesh_object(name, verts, faces)


def run(ob, mode="VOLUME", band=0.005):
    return compute.compute(ob, bpy.context.evaluated_depsgraph_get(), mode, band)


addon.register()
scene = bpy.context.scene
clear_scene()

# 1. Transformed box: the CoM follows location, volume follows scale.
ob = box("Box", (-1, -1, -1), (1, 1, 1))
ob.location = (1, 2, 3)
ob.rotation_euler = Euler((0.3, 0.7, -1.1))
ob.scale = (1, 2, 0.5)
r = run(ob)
check("box com", close(r.com, (1, 2, 3)), tuple(r.com))
# Blender stores vertices and matrices as float32: expect ~1e-7 relative error.
check("box volume", abs(r.volume - 8.0) < 1e-6 * 8.0, r.volume)
check("box clean", r.open_edges == 0 and r.nonmanifold_edges == 0 and not r.inconsistent_winding)

# 2. Negative scale flips winding: volume goes negative, CoM must not move.
ob.scale = (-1, 2, 0.5)
r = run(ob)
check("mirrored box com", close(r.com, (1, 2, 3)), tuple(r.com))
check("mirrored box volume sign", r.volume < 0 and not r.inconsistent_winding, r.volume)

# 3. Cone (any pyramid): solid CoM a quarter of the height above the base.
clear_scene()
bpy.ops.mesh.primitive_cone_add(vertices=64, radius1=1.0, depth=2.0, location=(0, 0, 0))
cone = bpy.context.active_object
r = run(cone)
check("cone solid com", close(r.com, (0, 0, -0.5)), tuple(r.com))
expected_tip = math.atan2(math.cos(math.pi / 64), 0.5)
check("cone stands", r.support_margin is not None and abs(r.support_margin - math.cos(math.pi / 64)) < 1e-6,
      r.support_margin)
check("cone tip angle", r.tip_angle is not None and abs(r.tip_angle - expected_tip) < 1e-6,
      f"{math.degrees(r.tip_angle or 0):.3f} vs {math.degrees(expected_tip):.3f}")
check("cone footprint", len(r.footprint) == 64 and abs(r.ground_z + 1.0) < 1e-6, len(r.footprint))

# Shell: lateral surface centroid at h/3 above the base, base disc at the base.
rs = run(cone, "SURFACE")
n = 64
apothem = math.cos(math.pi / n)
side = 2 * math.sin(math.pi / n)
base_area = 0.5 * n * side * apothem
slant = math.hypot(2.0, apothem)
lateral_area = 0.5 * n * side * slant
expected_z = (lateral_area * (-1 + 2 / 3) + base_area * -1) / (lateral_area + base_area)
check("cone shell com", close(rs.com, (0, 0, expected_z)), f"{rs.com.z:.6f} vs {expected_z:.6f}")

# 4. One flipped face is detected.
me = cone.data
bm = bmesh.new()
bm.from_mesh(me)
bm.faces.ensure_lookup_table()
bm.faces[0].normal_flip()
bm.to_mesh(me)
bm.free()
me.update()
r = run(cone)
check("flipped face detected", r.inconsistent_winding)

# 5. Open mesh.
clear_scene()
ob = box("Open", (0, 0, 0), (1, 1, 1))
bm = bmesh.new()
bm.from_mesh(ob.data)
bm.faces.ensure_lookup_table()
bm.faces.remove(bm.faces[1])
bm.to_mesh(ob.data)
bm.free()
r = run(ob)
check("open edges", r.open_edges == 4, r.open_edges)

# 6. Flat plane falls back to the surface model.
clear_scene()
bpy.ops.mesh.primitive_plane_add(size=2, location=(3, -1, 0.5))
r = run(bpy.context.active_object)
check("plane fallback", r.mode == "SURFACE" and bool(r.fallback), r.fallback)
check("plane com", close(r.com, (3, -1, 0.5)), tuple(r.com))

# 7. Modifiers are included: three cubes at x = 0, 2, 4.
clear_scene()
ob = box("Arrayed", (-1, -1, -1), (1, 1, 1))
mod = ob.modifiers.new("Array", "ARRAY")
mod.count = 3
mod.use_merge_vertices = False
r = run(ob)
check("array modifier com", close(r.com, (2, 0, 0)), tuple(r.com))
check("array modifier volume", abs(r.volume - 24.0) < 1e-6 * 24.0, r.volume)

# 8. Sheared prism: CoM at x = 2, footprint x in [0, 1] -> topples by 1.0.
clear_scene()
verts = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (3, 0, 4), (4, 0, 4), (4, 1, 4), (3, 1, 4)]
faces = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
ob = mesh_object("Leaning", verts, faces)
r = run(ob)
check("leaning com", close(r.com, (2.0, 0.5, 2.0)), tuple(r.com))
check("leaning topples", r.support_margin is not None and abs(r.support_margin + 1.0) < 1e-9, r.support_margin)

# 9. Vertex average is just that.
r = run(ob, "VERTICES")
check("vertex mean", close(r.com, (2.0, 0.5, 2.0)), tuple(r.com))

# 10. Irregular closed mesh against Blender's own Origin to Center of Mass (Volume).
clear_scene()
bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=4, radius=1.0, location=(0.4, -0.2, 1.3))
blob = bpy.context.active_object
rng = random.Random(7)
for v in blob.data.vertices:
    v.co *= 0.6 + 0.8 * rng.random()
    v.co.x *= 1.8
blob.data.update()
blob.rotation_euler = (0.2, -0.4, 0.9)
blob.scale = (1.3, 0.7, 1.1)
bpy.context.view_layer.update()
r = run(blob)
with bpy.context.temp_override(object=blob, active_object=blob,
                               selected_objects=[blob], selected_editable_objects=[blob]):
    bpy.ops.object.origin_set(type="ORIGIN_CENTER_OF_VOLUME")
bpy.context.view_layer.update()
blender_com = blob.matrix_world.translation
check("matches Blender volume centroid", close(r.com, blender_com, 1e-5),
      f"{tuple(round(c, 6) for c in r.com)} vs {tuple(round(c, 6) for c in blender_com)}")

# 11. Operators.
scene.com_viz.target = blob
bpy.ops.object.com_add_empty()
empty = bpy.data.objects.get(f"{blob.name}_CoM")
check("add empty", empty is not None and close(empty.location, r.com, 1e-5))

blob.location += Vector((5, 0, 0))
bpy.context.view_layer.update()
with bpy.context.temp_override(object=blob, active_object=blob, selected_objects=[blob],
                               selected_editable_objects=[blob]):
    bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="BOUNDS")
addon._results.clear()
bpy.ops.object.com_origin_to_com()
bpy.context.view_layer.update()
r2 = run(blob)
check("origin to com", close(blob.matrix_world.translation, r2.com, 1e-5),
      f"{tuple(blob.matrix_world.translation)} vs {tuple(r2.com)}")
check("origin to com keeps shape", abs(r2.volume - r.volume) < 1e-6 * abs(r.volume), f"{r2.volume} vs {r.volume}")

scene.cursor.location = (0, 0, 0)
bpy.ops.view3d.com_cursor_to_com()
check("cursor to com", close(scene.cursor.location, r2.com, 1e-5))

# 12. Units: 1 unit = 1 mm, a 10 mm PLA cube weighs 1.24 g.
clear_scene()
ob = box("Cube10", (0, 0, 0), (10, 10, 10))
scene.unit_settings.system = "METRIC"
scene.unit_settings.scale_length = 0.001
scene.com_viz.target = ob
scene.com_viz.material = "PLA"
r = run(ob)
mass = addon._mass_kg(r, scene.com_viz, scene)
check("mass in mm scene", abs(mass - 0.00124) < 1e-6 * 0.00124, f"{addon._fmt(scene, 'MASS', mass)}")
print("label:", addon._label(r, scene.com_viz, scene), "| length:", addon._fmt_length(scene, r.com.x))

addon.unregister()
addon.register()
addon.unregister()
check("register cycle", not hasattr(bpy.types.Scene, "com_viz"))

print(f"\n{len(failures)} failure(s): {failures}" if failures else "\nALL PASSED", flush=True)
if failures:
    sys.exit(1)
