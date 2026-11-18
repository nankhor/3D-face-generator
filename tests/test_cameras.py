import numpy as np

from facegen import landmarks as L
from facegen.cameras import anchor_key, build_observations, lm_key, model_points_from_canonical, solve_cameras
from facegen.geometry import look_at_camera, umeyama


def _scene(rng):
    V, _ = L.canonical_face()
    person = V * np.array([1.05, 0.97, 1.02]) + 0.003 * np.sin(V * 40)
    cams = {}
    for i, yaw in enumerate([-50, -25, 0, 25, 50]):
        a = np.radians(yaw)
        eye = np.array([0.45 * np.sin(a), 0.0, 0.45 * np.cos(a)])
        cams[f"p{i}.jpg"] = look_at_camera(eye, [0, 0, 0.03], [0, 1, 0], 3000, 4000, 3200)
    return person, cams


def test_solve_cameras_recovers_structure_and_focal():
    rng = np.random.default_rng(1)
    person, cams = _scene(rng)
    obs, sizes, focals = {}, {}, {}
    for name, cam in cams.items():
        px, _ = cam.project(person)
        obs[name] = {lm_key(j): px[j] + rng.normal(0, 1.5, 2) for j in range(468)}
        sizes[name] = (3000, 4000)
        focals[name] = 2900.0  # deliberately wrong
    sol = solve_cameras(obs, sizes, focals, model_points_from_canonical())
    assert not sol.failed
    assert max(sol.errors.values()) < 3.0
    keys = [lm_key(j) for j in range(468)]
    P = np.array([sol.points[k] for k in keys])
    sim = umeyama(P, person)
    rms = np.sqrt(np.mean(np.sum((sim.apply(P) - person) ** 2, axis=1)))
    assert rms < 0.001
    assert all(abs(c.focal - 3200) < 150 for c in sol.cameras.values())


def test_profile_photo_solved_from_clicked_anchors():
    rng = np.random.default_rng(2)
    person, cams = _scene(rng)
    ear = np.array([0.071, 0.005, -0.01])  # a point the landmark model does not have
    a = np.radians(88)
    cams["profile.jpg"] = look_at_camera(np.array([0.45 * np.sin(a), 0.0, 0.45 * np.cos(a)]), [0, 0, 0.03],
                                         [0, 1, 0], 3000, 4000, 3200)
    lms, anchors = [], {}
    for name, cam in cams.items():
        px, _ = cam.project(person)
        if name == "profile.jpg":
            lms.append(L.PhotoLandmarks(name, 3000, 4000, None))
            named = {"nose_tip": L.NOSE_TIP, "eye_outer_left": L.EYE_L_OUTER, "mouth_corner_left": L.MOUTH_L}
            anchors[name] = {k: px[i].tolist() for k, i in named.items()}
            anchors[name]["ear_left"] = cam.project(ear[None])[0][0].tolist()
        else:
            lms.append(L.PhotoLandmarks(name, 3000, 4000, px + rng.normal(0, 1.0, px.shape)))
            if name in ("p3.jpg", "p4.jpg"):
                anchors[name] = {"ear_left": cam.project(ear[None])[0][0].tolist()}
    obs = build_observations(lms, anchors)
    sizes = {n: (3000, 4000) for n in cams}
    focals = {n: 3200.0 for n in cams}
    sol = solve_cameras(obs, sizes, focals, model_points_from_canonical())
    assert "profile.jpg" in sol.cameras
    assert anchor_key("ear_left") in sol.points
    # the solved profile camera looks at the face from the side
    keys = [lm_key(j) for j in range(468)]
    sim = umeyama(np.array([sol.points[k] for k in keys]), person)
    centre = sim.apply(sol.cameras["profile.jpg"].center[None])[0]
    assert np.linalg.norm(centre - cams["profile.jpg"].center) < 0.03
