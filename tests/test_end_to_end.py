from facegen.selftest import run_selftest


def test_synthetic_pipeline(tmp_path):
    rep = run_selftest(tmp_path, log=lambda m: None)
    assert rep["passed"], rep
    assert rep["fit_rms_error_mm"] < 1.0
    assert rep["jaw_rms_error_mm"] < 0.5 * rep["template_vs_person_jaw_rms_mm"]
    assert rep["export_roundtrip_error_mm"] < 0.01
