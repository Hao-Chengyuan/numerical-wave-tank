"""
test_oceanwave3d.py — Validation tests for src/export/oceanwave3d.py.

Run standalone:
    .venv/bin/python src/tests/test_oceanwave3d.py
"""

import os
import pathlib
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
from tests import PASS, FAIL
from export.oceanwave3d import write_iwf
from wave_tank import IrregularWave_JONSWAP, RegularWave


def test_iwf_export_roundtrip():
    """write_iwf produces a 3-line header + N samples matching the wave model."""
    wave = IrregularWave_JONSWAP(Hs=2.0, Tp=8.0, d=30.0, seed=0)
    dt = 0.1
    t = np.arange(0.0, 60.0, dt)
    expected_eta = wave.surface_elevation(0.0, t)

    print(f"[1] OceanWave3D .iwf export round-trip")

    fd, tmp_name = tempfile.mkstemp(suffix=".iwf")
    os.close(fd)
    tmp_path = pathlib.Path(tmp_name)
    try:
        write_iwf(tmp_path, wave, x=0.0, t=t)
        lines = tmp_path.read_text().splitlines()
    finally:
        tmp_path.unlink(missing_ok=True)

    header_ok = lines[0].startswith("JONSWAP ")
    dt_ok = abs(float(lines[1]) - dt) < 1e-9
    n_ok = (len(lines) - 2) == len(t)
    first_ok = abs(float(lines[2]) - float(expected_eta[0])) < 5e-7

    ok = header_ok and dt_ok and n_ok and first_ok
    print(f"    header starts with 'JONSWAP ': {header_ok}")
    print(f"    dt match: {dt_ok}  (got {float(lines[1])})")
    print(f"    sample count: {n_ok}  (got {len(lines) - 2}, want {len(t)})")
    print(f"    first sample match: {first_ok}  (diff "
          f"{abs(float(lines[2]) - float(expected_eta[0])):.2e})")
    print(f"    {PASS if ok else FAIL}")
    return ok


def test_iwf_export_validates_inputs():
    """write_iwf raises TypeError / ValueError on invalid arguments."""
    wave = IrregularWave_JONSWAP(Hs=2.0, Tp=8.0, d=30.0, seed=0)
    regular = RegularWave(H=2.0, T=8.0, d=30.0)

    print(f"\n[2] OceanWave3D .iwf export input validation")

    fd, tmp_name = tempfile.mkstemp(suffix=".iwf")
    os.close(fd)
    tmp_path = pathlib.Path(tmp_name)
    os.unlink(tmp_path)

    cases = [
        ("RegularWave rejected", TypeError,
         lambda: write_iwf(tmp_path, regular, x=0.0,
                           t=np.arange(10.0) * 0.1)),
        ("2-D t rejected", ValueError,
         lambda: write_iwf(tmp_path, wave, x=0.0,
                           t=np.arange(10.0).reshape(2, 5))),
        ("t with < 2 samples rejected", ValueError,
         lambda: write_iwf(tmp_path, wave, x=0.0,
                           t=np.array([0.0]))),
        ("non-uniform t rejected", ValueError,
         lambda: write_iwf(tmp_path, wave, x=0.0,
                           t=np.array([0.0, 0.1, 0.3, 0.4]))),
    ]
    passed = []
    try:
        for label, exc_type, call in cases:
            try:
                call()
            except exc_type:
                passed.append(True)
                print(f"    {label}: raised {exc_type.__name__}")
            except Exception as e:
                passed.append(False)
                print(f"    {label}: raised wrong exception "
                      f"{type(e).__name__}")
            else:
                passed.append(False)
                print(f"    {label}: did not raise")

        no_file = not tmp_path.exists()
        passed.append(no_file)
        print(f"    no file created on error: {no_file}")
    finally:
        tmp_path.unlink(missing_ok=True)

    ok = all(passed)
    print(f"    {PASS if ok else FAIL}")
    return ok


TESTS = [
    test_iwf_export_roundtrip,
    test_iwf_export_validates_inputs,
]


if __name__ == "__main__":
    results = [t() for t in TESTS]
    print(f"\n{sum(results)}/{len(results)} tests passed.")
