"""
Smart Notes Assistant - Test Runner
Executes the automated test suite without requiring third-party test runners.
Can also be executed with: pytest tests/test_notes.py
"""

import unittest
import sys
import os

if __name__ == "__main__":
    test_loader = unittest.defaultTestLoader
    test_suite = test_loader.discover(start_dir="tests", pattern="test_*.py")
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    sys.exit(0 if result.wasSuccessful() else 1)
