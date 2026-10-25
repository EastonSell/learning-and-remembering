"""Expose the additive study's synthetic checks to the repository root suite."""
import importlib.util
from pathlib import Path
import unittest

STUDY = Path(__file__).resolve().parents[1] / 'studies' / 'pendigits-useful-compute'

def load_tests(loader, tests, pattern):
    suite = unittest.TestSuite()
    for filename in ('test_engine.py', 'test_runner.py', 'test_public_derivative.py'):
        spec = importlib.util.spec_from_file_location('useful_compute_'+filename[:-3], STUDY/filename)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        suite.addTests(loader.loadTestsFromModule(module))
    return suite
