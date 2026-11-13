import subprocess
import sys

def run_all():
    tests = [
        "test_rag.py",
        "test_agents.py",
        "test_memory.py",
        "test_router.py",
        "test_integration.py"
    ]
    results = {}

    print("========================================")
    print("RUNNING ALL 5 END-TO-END TEST SUITES")
    print("========================================")

    for test_file in tests:
        print(f"\n[RUNNING] {test_file}...")
        res = subprocess.run([sys.executable, test_file], capture_output=True, text=True)
        passed = (res.returncode == 0)
        results[test_file] = {
            "passed": passed,
            "code": res.returncode,
            "stdout": res.stdout,
            "stderr": res.stderr
        }
        status_str = "PASSED" if passed else "FAILED"
        print(f"[{status_str}] {test_file} (Exit code: {res.returncode})")
        if not passed:
            print("STDERR:", res.stderr[:300])

    print("\n========================================")
    print("FINAL SUMMARY REPORT")
    print("========================================")
    all_passed = True
    for test_file, data in results.items():
        status = "PASSED" if data["passed"] else "FAILED"
        print(f"  {test_file:<24}: {status}")
        if not data["passed"]:
            all_passed = False

    print("========================================")
    print("OVERALL STATUS:", "ALL 5 TEST SUITES PASSED" if all_passed else "SOME TESTS FAILED")
    print("========================================")

if __name__ == "__main__":
    run_all()
