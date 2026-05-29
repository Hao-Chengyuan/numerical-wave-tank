"""
fun_test.py — Orchestrator that runs every module's test suite in order.

Run:
    .venv/bin/python src/tests/fun_test.py

Individual module suites are also runnable standalone:
    .venv/bin/python src/tests/test_wave_tank.py
    .venv/bin/python src/tests/test_oceanwave3d.py
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from tests import test_oceanwave3d, test_wave_tank

MODULE_SUITES = [
    ("wave_tank", test_wave_tank.TESTS),
    ("oceanwave3d", test_oceanwave3d.TESTS),
]


def run_all():
    print("=" * 62)
    print("  fun_test.py — Validation & Performance Tests")
    print("=" * 62)

    results = []
    for name, tests in MODULE_SUITES:
        print(f"\n{'-' * 26} {name} {'-' * (34 - len(name))}")
        results.extend(t() for t in tests)

    print("\n" + "=" * 62)
    n_pass = sum(results)
    print(f"  {n_pass}/{len(results)} tests passed.")
    print("=" * 62)


if __name__ == "__main__":
    run_all()
