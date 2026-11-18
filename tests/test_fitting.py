import numpy as np

from facegen import landmarks as L
from facegen.fitting import FitSettings, JawCurve, LineView, fit_head
from facegen.geometry import look_at_camera
from facegen.head import HeadModel, embed_points
from facegen.regions import compute_regions


def test_jaw_lines_fix_the_jaw_under_a_beard():
    V, F = L.canonical_face()
    head = HeadModel(V, F, np.arange(468), *embed_points(V, F, V))
    person = V.copy()
    low = person[:, 1] < -0.03
    person[low, 0] *= 1.12
    regions = compute_regions(V, F, head.landmark_positions())
    targets = {i: person[i] for i in range(468)}
    X, _ = fit_head(head, targets, [], [], FitSettings(iterations=6), regions=regions)
    beard = regions["beard"] > 0.5
    err_without = np.sqrt(np.mean(np.sum((X - person) ** 2, 1)[beard]))
    jaw = JawCurve(head).positions(person, F)
    views = []
    for yaw in (0, 60, -60):
        a = np.radians(yaw)
        cam = look_at_camera(np.array([0.45 * np.sin(a), 0, 0.45 * np.cos(a)]), [0, 0, 0.03], [0, 1, 0], 2000, 2000, 2400)
        views.append(LineView(f"y{yaw}", cam, [cam.project(jaw)[0]]))
    X2, rep = fit_head(head, targets, [], views, FitSettings(iterations=8), regions=regions)
    err_with = np.sqrt(np.mean(np.sum((X2 - person) ** 2, 1)[beard]))
    assert err_with < 0.5 * err_without
    assert max(rep.line_rms_px.values()) < 3
