"""Driver tests use mocked training/scoring and synthetic fixtures only."""
from contextlib import nullcontext
import copy
import json
import os
from pathlib import Path
import resource
import signal
import tempfile
import time
import unittest
from unittest.mock import patch
import numpy as np
from functional_circuits import replay_study as study
from functional_circuits.replay_data import digest


def fixture():
    y = np.arange(10)
    x = np.zeros((10,64)); x[:,0] = y/10
    val, test = x.copy(), x.copy(); val[:,1] = .25; test[:,1] = .75
    return dict(train=(x,y,np.arange(10)+100), validation=(val,y), test=(test,y),
                identity={'synthetic_fixture':True})


def fake_train(x, y, ids, seed, rate, arm):
    return dict(arm=arm,seed=seed,rate=rate,label_order=list(range(10)),
                initial_sha256='same-initialization', counts={'presentations':80,'updates':20},
                snapshots=[[np.array([rate])] for _ in range(5)])


class ReplayStudyTests(unittest.TestCase):
    def setup_context(self):
        self.enterContext(patch.dict(os.environ,{k:'1' for k in study.THREAD_VARIABLES}))
        self.enterContext(patch.object(study,'limits',lambda seconds:nullcontext()))
        self.enterContext(patch.object(study,'peak_rss',return_value=20_000_000))
        self.loader = self.enterContext(patch.object(study,'load_verified_data',side_effect=lambda _:fixture()))
        self.trainer = self.enterContext(patch.object(study.core,'train_stream',side_effect=fake_train))

    def prediction(self, parameters, x):
        y = np.rint(x[:,0]*10).astype(int)
        if parameters[0][0] != .1:
            y[0] = 1
        return y

    def test_separate_stages_lock_before_test_and_all_prespecified_runs(self):
        self.setup_context()
        with tempfile.TemporaryDirectory() as tmp:
            dev, test = Path(tmp)/'dev', Path(tmp)/'test'
            def predict(parameters, x):
                if np.all(x[:,1] == .75):
                    self.assertTrue((test/'evaluation_lock.json').exists())
                else:
                    self.assertTrue(np.all(x[:,1] == .25))
                return self.prediction(parameters,x)
            predictor = self.enterContext(patch.object(study.core,'predict',side_effect=predict))
            sha = study.develop('synthetic',dev)
            self.assertEqual(predictor.call_count,18)
            self.assertTrue(all(np.all(c.args[1][:,1]==.25) for c in predictor.call_args_list))
            selection = study.read_selection(dev/'selection.json',sha)
            self.assertEqual(selection['selected_rate'],.1)
            self.assertEqual(len(selection['development_scores']),18)
            self.assertEqual(selection['test_model_evaluations'],0)
            result = study.evaluate('synthetic',dev/'selection.json',sha,test)
            self.assertEqual(self.trainer.call_count,48)
            self.assertEqual(predictor.call_count,18+30*5)
            rows = json.loads((test/'metrics.json').read_text())
            self.assertEqual({(r['seed'],r['model']) for r in rows},
                             {(s,a) for s in study.CONFIG['evaluation_seeds'] for a in study.CONFIG['arms']})
            self.assertEqual(result['primary_per_seed'],[0.]*10)
            self.assertFalse(result['acquisition_collapse'])
            for folder in (dev,test):
                m=json.loads((folder/'manifest.json').read_text())
                for name, value in m['artifact_sha256'].items():
                    self.assertEqual(digest((folder/name).read_bytes()),value)
            with self.assertRaises(FileExistsError):
                study.develop('synthetic',dev)

    def test_tampered_selection_rejected_before_data_or_test_access(self):
        self.setup_context()
        self.enterContext(patch.object(study.core,'predict',side_effect=self.prediction))
        with tempfile.TemporaryDirectory() as tmp:
            dev=Path(tmp)/'dev'; sha=study.develop('synthetic',dev)
            source=json.loads((dev/'selection.json').read_text())
            changes=[lambda d:d.update(selected_rate=.03),
                     lambda d:d['config'].update(evaluation_seeds=[211]),
                     lambda d:d['versions'].update(numpy='wrong'),
                     lambda d:d['source_sha256'].update(fake='wrong'),
                     lambda d:d['development_scores'].pop(),
                     lambda d:d.update(test_model_evaluations=1)]
            for i, change in enumerate(changes):
                altered=copy.deepcopy(source); change(altered)
                folder=Path(tmp)/f'bad-{i}'; folder.mkdir()
                p=folder/'selection.json'; p.write_text(json.dumps(altered))
                manifest=json.loads((dev/'manifest.json').read_text())
                manifest['artifact_sha256']['selection.json']=digest(p.read_bytes())
                (folder/'manifest.json').write_text(json.dumps(manifest))
                before=self.loader.call_count
                with self.assertRaises(ValueError):
                    study.evaluate('synthetic',p,digest(p.read_bytes()),Path(tmp)/f'out-{i}')
                self.assertEqual(self.loader.call_count,before)
            with self.assertRaises(ValueError):
                study.read_selection(dev/'selection.json','0'*64)

    def test_changed_data_identity_rejected_before_model_calls(self):
        self.setup_context()
        self.enterContext(patch.object(study.core,'predict',side_effect=self.prediction))
        with tempfile.TemporaryDirectory() as tmp:
            dev=Path(tmp)/'dev'; sha=study.develop('synthetic',dev)
            d=fixture(); d['identity']={'changed':True}
            self.loader.side_effect=lambda _:d
            before=self.trainer.call_count
            with self.assertRaises(ValueError):
                study.evaluate('synthetic',dev/'selection.json',sha,Path(tmp)/'test')
            self.assertEqual(self.trainer.call_count,before)

    def test_incomplete_development_cannot_be_evaluated(self):
        self.setup_context()
        self.enterContext(patch.object(study.core,'predict',side_effect=self.prediction))
        with tempfile.TemporaryDirectory() as tmp:
            dev=Path(tmp)/'dev'; sha=study.develop('synthetic',dev)
            (dev/'incomplete.json').write_text('{}')
            before=self.loader.call_count
            with self.assertRaises(ValueError):
                study.evaluate('synthetic',dev/'selection.json',sha,Path(tmp)/'test')
            self.assertEqual(self.loader.call_count,before)
            (dev/'incomplete.json').unlink()
            (dev/'manifest.json').unlink()
            with self.assertRaises(ValueError):
                study.read_selection(dev/'selection.json',sha)

    def test_cli_records_resource_failure_without_claiming_completion(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'failed'
            def fail(archive, output):
                output.mkdir()
                raise TimeoutError('fixture timeout')
            with patch('sys.argv',['run_replay_digits.py','develop','--archive','fixture','--output',str(out)]), \
                 patch.object(study,'develop',side_effect=fail), self.assertRaises(TimeoutError):
                study.main()
            r=json.loads((out/'incomplete.json').read_text())
            self.assertEqual(r['status'],'incomplete')
            self.assertFalse((out/'manifest.json').exists())

    def test_modified_published_audit_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'audit.json'; p.write_text('{}')
            with patch.object(study,'AUDIT',p), self.assertRaises(ValueError):
                study.load_verified_data('not-read')

    def test_thread_and_artifact_guards(self):
        with patch.dict(os.environ,{k:'1' for k in study.THREAD_VARIABLES}):
            study.thread_guard()
            with patch.dict(os.environ,{'OPENBLAS_NUM_THREADS':'2'}), self.assertRaises(ValueError):
                study.thread_guard()
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            with (p/'sparse').open('wb') as f:
                f.seek(250_000_000-1);f.write(b'0')
            with self.assertRaises(RuntimeError):
                study.save(p,'blocked.json',{'x':1})
            self.assertFalse((p/'blocked.json').exists())

    def test_wall_alarm_restores_process_limits(self):
        if study.platform.system() != 'Linux':
            self.skipTest('Linux-specific resource guard')
        old=resource.getrlimit(resource.RLIMIT_AS)
        handler=signal.getsignal(signal.SIGALRM)
        with self.assertRaises(TimeoutError):
            with study.limits(.01):
                time.sleep(.05)
        self.assertEqual(resource.getrlimit(resource.RLIMIT_AS),old)
        self.assertEqual(signal.getsignal(signal.SIGALRM),handler)
        self.assertEqual(signal.getitimer(signal.ITIMER_REAL)[0],0)


if __name__ == '__main__':
    unittest.main()
