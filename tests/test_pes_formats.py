import numpy as np
import pytest

from facegen.pes import ftex
from facegen.pes.face_folder import TemplateFace, export_face
from facegen.pes.fmdl import Fmdl
from facegen.pes.fpk import FpkFile, new_fpk
from facegen.pes.synthetic import build_fmdl, sphere_head, write_face_folder


def _mesh(scale=1.0, texture="face_bsm_alp.ftex"):
    V, F, UV = sphere_head(0.1 * scale, 10, 12)
    n = np.c_[V / np.linalg.norm(V, axis=1, keepdims=True), np.ones(len(V))]
    return dict(positions=V, faces=F, uv=UV, normals=n, texture=texture, material="skin")


def test_fpk_roundtrip_and_relink():
    fpk = new_fpk({"/Assets/x/real/12345/face_high.fmdl": b"abc", "/Assets/x/real/12345/face_diff.bin": b"FACE"})
    back = FpkFile.from_bytes(fpk.to_bytes())
    assert [e.path for e in back.entries] == [e.path for e in fpk.entries]
    assert back.find("face_diff.bin").data == b"FACE"
    back.relink("/12345/", "/7654321/")
    again = FpkFile.from_bytes(back.to_bytes())
    assert again.entries[0].path == "/Assets/x/real/7654321/face_high.fmdl"
    assert again.entries[0].name_hash_matches()


def test_fmdl_parse_and_in_place_patch():
    m = _mesh()
    data = build_fmdl([m])
    f = Fmdl.parse(data)
    assert f.meshes[0].vertex_count == len(m["positions"])
    np.testing.assert_allclose(f.meshes[0].positions, m["positions"], atol=1e-6)
    np.testing.assert_array_equal(f.meshes[0].faces, m["faces"])
    new = f.meshes[0].positions * 1.1
    f.set_positions(0, new)
    f.grow_bounding_boxes(0.01)
    out = f.to_bytes()
    assert len(out) == len(data)  # in place: nothing else moves
    g = Fmdl.parse(out)
    np.testing.assert_allclose(g.meshes[0].positions, new, atol=1e-6)
    # every byte outside the position buffer is unchanged
    changed = np.nonzero(np.frombuffer(out, np.uint8) != np.frombuffer(data, np.uint8))[0]
    pos_bytes = set((f.meshes[0].position_offsets[:, None] + np.arange(12)).ravel().tolist())
    bbox_bytes = set(range(f.bbox_offsets[0], f.bbox_offsets[0] + 32))
    assert set(changed.tolist()) <= pos_bytes | bbox_bytes


@pytest.mark.parametrize("new_id", ["54321", "7654321"])
def test_fmdl_relink_same_and_different_length(new_id):
    data = build_fmdl([_mesh()])
    f = Fmdl.parse(data)
    assert f.relink("/12345/", f"/{new_id}/") == 1
    g = Fmdl.parse(f.to_bytes())
    assert any(f"/real/{new_id}/" in s for s in g.strings)
    assert not any("/12345/" in s for s in g.strings)
    np.testing.assert_allclose(g.meshes[0].positions, f.meshes[0].positions)


def test_ftex_roundtrip():
    img = np.zeros((128, 64, 4), np.uint8)
    img[..., 0] = 200
    img[..., 1] = np.linspace(0, 255, 64)[None]
    img[..., 3] = 255
    data = ftex.encode(img)
    info = ftex.read_info(data)
    assert (info.width, info.height, info.pixel_format) == (64, 128, 4)
    back = ftex.decode(data)
    assert np.abs(back.astype(int) - img.astype(int)).mean() < 4


def test_face_folder_export(tmp_path):
    folder = write_face_folder(tmp_path, "12345", [_mesh()], hair_meshes=[_mesh(1.05, "hair_parts_bsm_alp.ftex")])
    t = TemplateFace.open(folder)
    head, smap, _, _, _ = t.skin()
    fitted = head.vertices * np.array([1.08, 1.0, 1.0])
    res = export_face(t, head, smap, fitted, tmp_path / "out", "7654321", texture_rgba=t.base_texture())
    back = TemplateFace.open(res.folder)
    bh, _, _, _, _ = back.skin()
    np.testing.assert_allclose(np.sort(bh.vertices, 0), np.sort(fitted, 0), atol=1e-6)
    assert all("/7654321/" in e.path for e in back.fpk.entries)
    # hair moved with the skin; eyes moved outward with the wider head
    assert np.abs(back.hair.meshes[0].positions[:, 0]).max() > np.abs(t.hair.meshes[0].positions[:, 0]).max()
    assert abs(back.eyes()["eye_left"][0]) > abs(t.eyes()["eye_left"][0])
