"""Synthetic saved-JSON comparison probes. No research computation."""
import importlib.util
import copy
import json
import shutil
import tempfile
import unittest
from pathlib import Path
import portable
from jointaudit.admission import Admission

ROOT=Path(__file__).resolve().parents[1]

class ComparisonTests(unittest.TestCase):
    def subject(self):
        path=ROOT/'compare_reproduction.py'
        self.assertTrue(path.is_file(),'saved-science comparison module is missing')
        spec=importlib.util.spec_from_file_location('compare_subject',path)
        mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
        return mod

    def test_timing_differences_do_not_change_scientific_summary_comparison(self):
        mod=self.subject()
        a={'summaries':{'mean':1.5},'resources_sampled_before_summary_write':{'cpu':1}}
        b={'summaries':{'mean':1.5},'resources_sampled_before_summary_write':{'cpu':3}}
        self.assertTrue(mod.same_scientific_json(a,b,summary=True))
        self.assertFalse(mod.same_scientific_json(a,b,summary=False))

    def test_any_scientific_difference_fails_exact_comparison(self):
        mod=self.subject()
        self.assertFalse(mod.same_scientific_json({'loss':1.0},{'loss':1.000000001}))
        self.assertFalse(mod.same_scientific_json({'summaries':{'mean':1}},{'summaries':{'mean':2}},summary=True))
        self.assertFalse(mod.same_scientific_json({'state':'a'},{'state':'a','extra':1}))

    def synthetic_run(self,base,*,freeze_mutator=None,missing_freeze=False):
        # These are copied saved JSON, NOT model outputs produced by this test.
        # A consistent receipt must never be presented as observed execution.
        run=Path(base)/'SYNTHETIC-NOT-A-SCIENTIFIC-RUN'
        admission=Admission(run);admission.reserve_output()
        for source in (ROOT/'expected/diagnostic-v2').glob('*.json'):
            shutil.copyfile(source,run/source.name)
        self.assertTrue(callable(getattr(portable,'expected_freeze',None)), 'exact expected-freeze schema builder is missing')
        freeze=portable.expected_freeze()
        if freeze_mutator is not None: freeze=freeze_mutator(copy.deepcopy(freeze))
        if not missing_freeze: (run/'run-input-freeze.json').write_text(json.dumps(freeze,sort_keys=True,indent=2)+'\n')
        admission.finalize_success(states=16,seed_clusters=4)
        return run

    def comparison(self,run,exit_code=0):
        mod=self.subject()
        self.assertTrue(callable(getattr(mod,'compare_run',None)), 'provenance-validating run comparator is missing')
        return mod.compare_run(run,exit_code=exit_code)

    def test_complete_exact_synthetic_freeze_is_validated_without_claiming_execution_observation(self):
        with tempfile.TemporaryDirectory() as td:
            result=self.comparison(self.synthetic_run(td))
            self.assertTrue(result['exact_scientific_json_match'])
            self.assertTrue(result['completion_receipt_integrity_valid'])
            self.assertTrue(result['run_input_freeze_matches_current_package_and_training_catalog'])
            self.assertEqual(result['source_input_binding_count'],82)
            self.assertEqual(result['supplied_exit_code'],0)
            self.assertIs(result['execution_observed_by_comparator'],False)
            self.assertNotIn('new_completion_and_exit_valid',result)

    def test_empty_missing_and_wrong_source_freezes_reject_even_with_consistent_receipt(self):
        def changed_file(f,name):
            f['files'][name]='0'*64;return f
        variants={
            'empty':lambda f:{},
            'missing_binding':lambda f:dict(f,files={k:v for k,v in f['files'].items() if k!='PACKAGE-FILES.json'}),
            'extra_binding':lambda f:dict(f,files=dict(f['files'],unexpected='0'*64)),
            'wrong_package':lambda f:changed_file(f,'PACKAGE-FILES.json'),
            'wrong_train':lambda f:changed_file(f,'training/train-images-idx3-ubyte.gz'),
            'wrong_seed':lambda f:dict(f,seeds=[2001,2002,2003,2005]),
            'wrong_caps':lambda f:dict(f,caps=dict(f['caps'],threads=2)),
            'wrong_schema':lambda f:dict(f,format='wrong'),
            'wrong_type':lambda f:dict(f,scientific_execution_performed=0),
        }
        for name,mutator in variants.items():
            with self.subTest(variant=name),tempfile.TemporaryDirectory() as td:
                run=self.synthetic_run(td,freeze_mutator=mutator)
                with self.assertRaises(ValueError): self.comparison(run)
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(ValueError): self.comparison(self.synthetic_run(td,missing_freeze=True))

    def rewrite_receipt(self,run,**changes):
        receipt=json.loads((run/'COMPLETE.json').read_text());receipt.update(changes)
        total=sum(f.stat().st_size for f in run.iterdir() if f.is_file() and f.name!='COMPLETE.json')
        for _ in range(10):
            encoded=(json.dumps(receipt,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
            expected=total+len(encoded)
            if expected==receipt['artifact_bytes_including_complete']: break
            receipt['artifact_bytes_including_complete']=expected
        else: self.fail('synthetic completion byte fixed point failed')
        (run/'COMPLETE.json').write_bytes(encoded)

    def test_wrong_completion_status_rejects_despite_valid_hashes_and_byte_accounting(self):
        with tempfile.TemporaryDirectory() as td:
            run=self.synthetic_run(td)
            self.rewrite_receipt(run,status='not_a_success')
            with self.assertRaises(ValueError): self.comparison(run)

    def test_extra_directory_and_receipt_bound_extra_file_reject(self):
        for kind in ['directory','file']:
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as td:
                run=self.synthetic_run(td)
                if kind=='directory': (run/'unexpected').mkdir()
                else:
                    (run/'unexpected').write_bytes(b'synthetic')
                    receipt=json.loads((run/'COMPLETE.json').read_text())
                    receipt['files']['unexpected']={'bytes':9,'sha256':portable.sha256_file(run/'unexpected')}
                    self.rewrite_receipt(run,files=receipt['files'])
                with self.assertRaises(ValueError): self.comparison(run)

    def test_nonzero_supplied_exit_never_passes(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(ValueError): self.comparison(self.synthetic_run(td),exit_code=7)

if __name__=='__main__': unittest.main()
