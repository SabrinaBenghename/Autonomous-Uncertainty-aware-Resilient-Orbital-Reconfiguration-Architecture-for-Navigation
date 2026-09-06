import os
import sys
import traceback

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tests import test_dynamics, test_sensors, test_navigation, test_faults, test_ai


def run_test_module(module):
    print(f"\n--- Running tests in {module.__name__} ---")
    functions = [getattr(module, attr) for attr in dir(module) if attr.startswith("test_") and callable(getattr(module, attr))]
    passed = 0
    failed = 0
    for func in functions:
        try:
            func()
            print(f"  [PASS] {func.__name__}")
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {func.__name__}: {e}")
            traceback.print_exc()
            failed += 1
    return passed, failed


def main():
    modules = [test_dynamics, test_sensors, test_navigation, test_faults, test_ai]
    total_passed = 0
    total_failed = 0

    for mod in modules:
        p, f = run_test_module(mod)
        total_passed += p
        total_failed += f

    print("\n==========================================================")
    print(f"TEST RESULTS SUMMARY: {total_passed} Passed, {total_failed} Failed")
    print("==========================================================")

    if total_failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
