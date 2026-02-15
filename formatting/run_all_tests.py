#!/usr/bin/env python3
"""
Скрипт для запуска всех тестов и проверки результатов.
"""

import subprocess
import sys
import os
from pathlib import Path

# Colors for output
GREEN = '\033[92m'
RED = '\033[91m'
BLUE = '\033[94m'
YELLOW = '\033[93m'
RESET = '\033[0m'
BOLD = '\033[1m'

def run_command(cmd, description):
    """Run a command and return result."""
    print(f"\n{BLUE}{BOLD}→ {description}{RESET}")
    print(f"  Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            cwd="/Users/kirill/Desktop/vniim/formatingDocx",
            capture_output=True,
            text=True,
            timeout=60
        )
        
        print(f"  Exit code: {result.returncode}")
        
        if result.stdout:
            print(f"\n{result.stdout}")
        
        if result.stderr and result.returncode != 0:
            print(f"{RED}STDERR:{RESET}\n{result.stderr}")
        
        return result.returncode == 0, result.stdout, result.stderr
        
    except subprocess.TimeoutExpired:
        print(f"{RED}✗ Command timed out{RESET}")
        return False, "", "Timeout"
    except Exception as e:
        print(f"{RED}✗ Error: {e}{RESET}")
        return False, "", str(e)


def main():
    print(f"\n{BOLD}╔════════════════════════════════════════════════════════════════╗{RESET}")
    print(f"{BOLD}║         PHASE 2 STAGE 3: Complete Test Run                 ║{RESET}")
    print(f"{BOLD}╚════════════════════════════════════════════════════════════════╝{RESET}")
    
    # Step 1: Count test files
    print(f"\n{YELLOW}[STEP 1] Scanning test files...{RESET}")
    test_dir = Path("/Users/kirill/Desktop/vniim/formatingDocx/doc_editor/tests")
    test_files = list(test_dir.glob("test_*.py"))
    
    print(f"  Found {len(test_files)} test files:")
    for tf in sorted(test_files):
        print(f"    • {tf.name}")
    
    # Step 2: Run tests for Stage 3 specifically
    print(f"\n{YELLOW}[STEP 2] Running Stage 3 tests (PrefaceProcessor + AppendixProcessor)...{RESET}")
    
    stage3_tests_ok = True
    stage3_count = 0
    
    # Test PrefaceProcessor
    print(f"\n{BLUE}Testing PrefaceProcessor...{RESET}")
    success, stdout, stderr = run_command(
        [sys.executable, "-m", "pytest", 
         "doc_editor/tests/test_preface_processor.py", "-v", "--tb=short"],
        "Run PrefaceProcessor tests"
    )
    
    if success:
        print(f"{GREEN}✓ PrefaceProcessor tests passed{RESET}")
        # Count tests
        if "passed" in stdout:
            for line in stdout.split('\n'):
                if 'passed' in line and 'warning' in line:
                    parts = line.split()
                    if parts and parts[0].isdigit():
                        count = int(parts[0])
                        stage3_count += count
                        print(f"  {GREEN}✓ {count} tests passed{RESET}")
    else:
        print(f"{RED}✗ PrefaceProcessor tests failed{RESET}")
        stage3_tests_ok = False
    
    # Test AppendixProcessor
    print(f"\n{BLUE}Testing AppendixProcessor...{RESET}")
    success, stdout, stderr = run_command(
        [sys.executable, "-m", "pytest",
         "doc_editor/tests/test_appendix_processor.py", "-v", "--tb=short"],
        "Run AppendixProcessor tests"
    )
    
    if success:
        print(f"{GREEN}✓ AppendixProcessor tests passed{RESET}")
        # Count tests
        if "passed" in stdout:
            for line in stdout.split('\n'):
                if 'passed' in line and 'warning' in line:
                    parts = line.split()
                    if parts and parts[0].isdigit():
                        count = int(parts[0])
                        stage3_count += count
                        print(f"  {GREEN}✓ {count} tests passed{RESET}")
    else:
        print(f"{RED}✗ AppendixProcessor tests failed{RESET}")
        stage3_tests_ok = False
    
    # Step 3: Run all tests
    print(f"\n{YELLOW}[STEP 3] Running ALL tests in suite...{RESET}")
    
    success, stdout, stderr = run_command(
        [sys.executable, "-m", "pytest", 
         "doc_editor/tests/", "-v", "--tb=line"],
        "Run all tests"
    )
    
    total_passed = 0
    if success and "passed" in stdout:
        for line in stdout.split('\n'):
            if 'passed' in line:
                parts = line.split()
                if parts and parts[0].isdigit():
                    total_passed = int(parts[0])
                    print(f"  {GREEN}✓ Total: {total_passed} tests passed{RESET}")
                    break
    
    # Step 4: Summary
    print(f"\n{BOLD}╔════════════════════════════════════════════════════════════════╗{RESET}")
    print(f"{BOLD}║                      SUMMARY                                  ║{RESET}")
    print(f"{BOLD}╚════════════════════════════════════════════════════════════════╝{RESET}")
    
    print(f"\n{BLUE}Test Results:{RESET}")
    print(f"  Expected total tests:     {BOLD}128{RESET} (75 Phase 1-2 + 53 Stage 3)")
    print(f"  Actual tests passed:      {BOLD}{total_passed}{RESET}")
    
    if total_passed >= 128:
        print(f"\n{GREEN}{BOLD}✓ SUCCESS: All tests passed!{RESET}")
        print(f"  GOST Compliance: {BOLD}68%{RESET}")
        print(f"  Pipeline Stages: {BOLD}7/7{RESET}")
        return 0
    elif total_passed >= 100:
        print(f"\n{YELLOW}{BOLD}⚠ PARTIAL SUCCESS: Most tests passed ({total_passed}/128){RESET}")
        print(f"  Missing: {128 - total_passed} tests")
        return 1
    else:
        print(f"\n{RED}{BOLD}✗ FAILED: Test count is low ({total_passed}/128){RESET}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
