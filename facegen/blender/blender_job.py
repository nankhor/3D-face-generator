"""Runs *inside* Blender (2.93 or newer): builds a scene from a facegen job and renders it.

Usage (done by facegen): blender -b --factory-startup --python blender_job.py -- job.json

Keep this file compatible with Python 3.9 and Blender's bundled numpy.

Job JSON:
    {
      "meshes": [{"name": "head", "npz": "...", "texture": "....png" | null}],
      "cameras": [{"name": "...", "width": W, "height": H, "K": [[..]], "R": [[..]], "t": [..],
                   "background": "photo.jpg" | null}],
      "style": "textured" | "clay",
      "renders_dir": "..." | null,      # render every camera here
      "save_blend": "....blend" | null, # save the scene (for opening in the Blender UI)
      "samples": 16
    }
"""

import json
import math
import os
import sys

import bpy
import numpy as np
from mathutils import Matrix

# head space (y up, z forward) -> Blender (z up, -y forward)
HEAD_TO_BLENDER = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
CV_TO_BLENDER_CAM = np.diag([1.0, -1.0, -1.0])


def clear_scene():
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    for coll in (bpy.data.meshes, bpy.data.materials, bpy.data.images, bpy.data.cameras, bpy.data.lights):
        for item in list(coll):
            coll.remove(item)


def make_material(name, texture_path, style):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    for n in list(nodes):
        nodes.remove(n)
    out = nodes.new("ShaderNodeOutputMaterial")
    if style == "textured" and texture_path and os.path.exists(texture_path):
        tex = nodes.new("ShaderNodeTexImage")
        tex.image = bpy.data.images.load(texture_path)
        emit = nodes.new("ShaderNodeEmission")
        links.new(tex.outputs["Color"], emit.inputs["Color"])
        links.new(emit.outputs["Emission"], out.inputs["Surface"])
    else:
        bsdf = nodes.new("ShaderNodeBsdfDiffuse")
        bsdf.inputs["Color"].default_value = (0.75, 0.68, 0.62, 1.0)
        links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    return mat


def add_mesh(spec, style):
    data = np.load(spec["npz"])
    V = data["vertices"] @ HEAD_TO_BLENDER.T
    F = data["faces"].astype(np.int64)
    mesh = bpy.data.meshes.new(spec["name"])
    mesh.from_pydata(V.tolist(), [], F.tolist())
    if "uvs" in data.files:
        uvs = data["uvs"]
        layer = mesh.uv_layers.new(name="UVMap")
        loop_vi = np.zeros(len(mesh.loops), dtype=np.int64)
        mesh.loops.foreach_get("vertex_index", loop_vi)
        loop_uv = uvs[loop_vi].copy()
        loop_uv[:, 1] = 1.0 - loop_uv[:, 1]
        layer.data.foreach_set("uv", loop_uv.ravel())
    for p in mesh.polygons:
        p.use_smooth = True
    mesh.update()
    obj = bpy.data.objects.new(spec["name"], mesh)
    bpy.context.scene.collection.objects.link(obj)
    obj.data.materials.append(make_material(spec["name"] + "_mat", spec.get("texture"), style))
    return obj


def add_camera(spec):
    w, h = int(spec["width"]), int(spec["height"])
    K = np.array(spec["K"], dtype=float)
    R = np.array(spec["R"], dtype=float)
    t = np.array(spec["t"], dtype=float)
    cam_data = bpy.data.cameras.new(spec["name"])
    horizontal = w >= h
    cam_data.sensor_fit = "HORIZONTAL" if horizontal else "VERTICAL"
    cam_data.sensor_width = 36.0
    cam_data.sensor_height = 36.0
    size = w if horizontal else h
    cam_data.lens = K[0, 0] * 36.0 / size
    longest = float(max(w, h))
    cam_data.shift_x = -(K[0, 2] - w / 2.0) / longest
    cam_data.shift_y = (K[1, 2] - h / 2.0) / longest
    cam_data.clip_start = 0.01
    cam_data.clip_end = 100.0
    obj = bpy.data.objects.new(spec["name"], cam_data)
    bpy.context.scene.collection.objects.link(obj)
    centre = -R.T @ t
    rot = HEAD_TO_BLENDER @ R.T @ CV_TO_BLENDER_CAM
    loc = HEAD_TO_BLENDER @ centre
    m = np.eye(4)
    m[:3, :3] = rot
    m[:3, 3] = loc
    obj.matrix_world = Matrix(m.tolist())  # rows
    bg = spec.get("background")
    if bg and os.path.exists(bg):
        cam_data.show_background_images = True
        img = cam_data.background_images.new()
        img.image = bpy.data.images.load(bg)
        img.alpha = 0.6
    obj["facegen_size"] = [w, h]
    return obj


def setup_render(style, samples):
    scene = bpy.context.scene
    if style == "clay":
        scene.render.engine = "BLENDER_WORKBENCH"
        try:
            scene.display.shading.light = "STUDIO"
            scene.display.shading.color_type = "MATERIAL"
        except Exception:
            pass
    else:
        scene.render.engine = "CYCLES"
        scene.cycles.samples = samples
        try:
            scene.cycles.use_denoising = False
        except Exception:
            pass
        world = scene.world or bpy.data.worlds.new("World")
        scene.world = world
        world.use_nodes = True
        bgnode = world.node_tree.nodes.get("Background")
        if bgnode:
            bgnode.inputs[0].default_value = (1.0, 1.0, 1.0, 1.0)
            bgnode.inputs[1].default_value = 1.0
    scene.render.image_settings.file_format = "PNG"
    scene.render.film_transparent = False
    try:
        scene.view_settings.view_transform = "Standard"
    except Exception:
        pass


def run_job(job):
    clear_scene()
    style = job.get("style", "textured")
    for spec in job.get("meshes", []):
        add_mesh(spec, style)
    if style == "clay":
        light = bpy.data.lights.new("key", "SUN")
        lobj = bpy.data.objects.new("key", light)
        bpy.context.scene.collection.objects.link(lobj)
        lobj.rotation_euler = (math.radians(50), 0, math.radians(30))
    cams = [add_camera(c) for c in job.get("cameras", [])]
    setup_render(style, int(job.get("samples", 16)))
    outputs = []
    renders_dir = job.get("renders_dir")
    if renders_dir:
        os.makedirs(renders_dir, exist_ok=True)
        scene = bpy.context.scene
        max_side = int(job.get("max_side", 800))
        for cam in cams:
            w, h = cam["facegen_size"]
            f = min(1.0, max_side / float(max(w, h)))
            scene.render.resolution_x = int(round(w * f))
            scene.render.resolution_y = int(round(h * f))
            scene.render.resolution_percentage = 100
            scene.camera = cam
            path = os.path.join(renders_dir, cam.name + ".png")
            scene.render.filepath = path
            bpy.ops.render.render(write_still=True)
            outputs.append(path)
    if cams:
        bpy.context.scene.camera = cams[0]
    if job.get("save_blend"):
        bpy.ops.wm.save_as_mainfile(filepath=job["save_blend"])
        outputs.append(job["save_blend"])
    return outputs


def main():
    argv = sys.argv
    args = argv[argv.index("--") + 1:] if "--" in argv else []
    if not args:
        print("FACEGEN_ERROR no job file given")
        return
    with open(args[0]) as fh:
        job = json.load(fh)
    try:
        outputs = run_job(job)
        print("FACEGEN_DONE " + json.dumps(outputs))
    except Exception as exc:  # report and fail
        import traceback

        traceback.print_exc()
        print("FACEGEN_ERROR " + str(exc))
        sys.exit(1)


if __name__ == "__main__":
    main()
