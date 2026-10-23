"""Synthetic-only portable binding tests. No retained arrays are decoded."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]

class PortableTests(unittest.TestCase):
    def receipt_recipe(self):
        text=(ROOT/'README.md').read_text()
        begin='# BEGIN EXCLUSIVE RECEIPTS';end='# END EXCLUSIVE RECEIPTS'
        self.assertIn(begin,text,'exclusive receipt recipe is missing')
        self.assertIn(end,text)
        block=text.split(begin,1)[1].split(end,1)[0]
        lines=[line[4:] if line.startswith('    ') else line for line in block.splitlines()]
        commands=[line for line in lines if line.lstrip().startswith('env OPENBLAS_NUM_THREADS=1 ')]
        self.assertEqual(len(commands),1,'exactly one explicit scientific command must be replaceable before this synthetic shell test')
        fake='(printf invoked > "$work/toy-invoked"; printf "TOY STDOUT\\n"; printf "TOY STDERR\\n" >&2; exit 7) >&3 2>&4 || code=$?'
        return '\n'.join(lines).replace(commands[0],fake)

    def test_documented_receipts_preserve_each_existing_target_and_do_not_invoke_command(self):
        script='set -u\nset -C\nwork="$1"\n'+self.receipt_recipe()
        for leaf in ['stdout.txt','stderr.txt','exit-code.txt']:
            with self.subTest(leaf=leaf),tempfile.TemporaryDirectory() as td:
                root=Path(td);sentinel=root/leaf;sentinel.write_text('PRESERVE ME')
                result=subprocess.run(['sh','-c',script,'synthetic-receipts',str(root)],capture_output=True,text=True,timeout=5)
                self.assertNotEqual(result.returncode,0)
                self.assertEqual(sentinel.read_text(),'PRESERVE ME')
                self.assertFalse((root/'toy-invoked').exists())

    def test_documented_receipts_capture_actual_nonzero_toy_exit_without_substitution(self):
        script='set -u\nset -C\nwork="$1"\n'+self.receipt_recipe()+'\nexit "$code"\n'
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            result=subprocess.run(['sh','-c',script,'synthetic-receipts',str(root)],capture_output=True,text=True,timeout=5)
            self.assertEqual(result.returncode,7,result.stderr)
            self.assertEqual((root/'stdout.txt').read_text(),'TOY STDOUT\n')
            self.assertEqual((root/'stderr.txt').read_text(),'TOY STDERR\n')
            self.assertEqual((root/'exit-code.txt').read_text(),'7\n')

    def test_acquisition_recipe_stops_when_fresh_directory_creation_fails(self):
        text=(ROOT/'ACQUIRE-TRAINING-DATA.md').read_text()
        begin='# BEGIN FAIL-CLOSED TRAIN ACQUISITION';end='# END FAIL-CLOSED TRAIN ACQUISITION'
        self.assertIn(begin,text,'fail-closed training acquisition block is missing')
        block=text.split(begin,1)[1].split(end,1)[0]
        lines=[line[4:] if line.startswith('    ') else line for line in block.splitlines()]
        script='\n'.join(lines)
        self.assertEqual(script.count('train=/absolute/TRAIN'),1)
        script=script.replace('train=/absolute/TRAIN','train="$1"')
        # An explicit fake prevents network even if the setup guard regresses.
        script='curl() { printf invoked > "$train/toy-curl-invoked"; return 99; }\n'+script
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);sentinel=root/'train-images-idx3-ubyte.gz';sentinel.write_text('PRESERVE ME')
            result=subprocess.run(['sh','-c',script,'synthetic-acquisition',str(root)],capture_output=True,text=True,timeout=5)
            self.assertNotEqual(result.returncode,0)
            self.assertEqual(sentinel.read_text(),'PRESERVE ME')
            self.assertFalse((root/'toy-curl-invoked').exists())

    def load_portable(self):
        path=ROOT/'portable.py'
        self.assertTrue(path.is_file(), 'portable scientific-input binding module is missing')
        spec=importlib.util.spec_from_file_location('portable_test_subject',path)
        mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
        return mod

    def test_safe_relative_bindings_reject_traversal_and_symlinks(self):
        p=self.load_portable()
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); (root/'good').write_bytes(b'x')
            self.assertEqual(p.safe_file(root,'good'),root/'good')
            for value in ('../good','/good','a/../good','./good'):
                with self.assertRaises(ValueError): p.safe_file(root,value)
            (root/'link').symlink_to(root/'good')
            with self.assertRaises(ValueError): p.safe_file(root,'link')
            (root/'dirlink').symlink_to(root,target_is_directory=True)
            with self.assertRaises(ValueError): p.safe_file(root,'dirlink/good')

    def test_hash_only_catalog_detects_missing_corrupt_and_extra_inputs(self):
        p=self.load_portable()
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); (root/'inputs').mkdir(); (root/'inputs/a').write_bytes(b'synthetic')
            catalog={'files':{'inputs/a':{'sha256':p.sha256_file(root/'inputs/a'),'bytes':9}}}
            got=p.verify_packaged_inputs(root,catalog)
            self.assertEqual(got,{'inputs/a':catalog['files']['inputs/a']['sha256']})
            (root/'inputs/b').write_bytes(b'extra')
            with self.assertRaises(ValueError): p.verify_packaged_inputs(root,catalog)
            (root/'inputs/b').unlink(); (root/'inputs/a').write_bytes(b'changed')
            with self.assertRaises(ValueError): p.verify_packaged_inputs(root,catalog)
            (root/'inputs/a').unlink()
            with self.assertRaises(ValueError): p.verify_packaged_inputs(root,catalog)

    def test_package_manifest_named_extra_is_not_ignored(self):
        p=self.load_portable()
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); (root/'inputs').mkdir(); (root/'inputs/a').write_bytes(b'a')
            catalog={'files':{'inputs/a':{'sha256':p.sha256_file(root/'inputs/a'),'bytes':1}}}
            (root/'input-catalog.json').write_text(json.dumps(catalog))
            files={n:{'sha256':p.sha256_file(root/n),'bytes':(root/n).stat().st_size} for n in ['inputs/a','input-catalog.json']}
            (root/'PACKAGE-FILES.json').write_text(json.dumps({'files':files}))
            p.verify_package(root)
            (root/'extra').mkdir(); (root/'extra/PACKAGE-FILES.json').write_text('{}')
            with self.assertRaises(ValueError): p.verify_package(root)

    def test_training_hash_check_uses_compressed_bytes_without_decode(self):
        p=self.load_portable()
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); (root/'synthetic.gz').write_bytes(b'not a gzip, just a hash fixture')
            info={'synthetic.gz':{'sha256':p.sha256_file(root/'synthetic.gz'),'bytes':(root/'synthetic.gz').stat().st_size}}
            self.assertEqual(p.verify_training(root,info),{'training/synthetic.gz':info['synthetic.gz']['sha256']})
            (root/'synthetic.gz').write_bytes(b'changed')
            with self.assertRaises(ValueError): p.verify_training(root,info)

    def test_package_check_does_not_import_numpy(self):
        self.load_portable()
        code="import sys; sys.path.insert(0,sys.argv[1]); import portable; portable.verify_package(); assert 'numpy' not in sys.modules"
        result=subprocess.run([sys.executable,'-B','-c',code,str(ROOT)],capture_output=True,text=True,timeout=20)
        self.assertEqual(result.returncode,0,result.stderr)

    def test_frozen_bindings_are_role_relative_and_relocation_safe(self):
        p=self.load_portable()
        with tempfile.TemporaryDirectory() as td:
            base=Path(td); package=base/'package'; package.mkdir()
            (package/'inputs').mkdir(); (package/'inputs/a').write_bytes(b'a')
            raw=base/'raw'; raw.mkdir(); (raw/'train.gz').write_bytes(b'z')
            frozen={'files':{'inputs/a':p.sha256_file(package/'inputs/a'),'training/train.gz':p.sha256_file(raw/'train.gz')}}
            p.verify_frozen(frozen,package,raw)
            relocated=base/'moved'; package.rename(relocated)
            p.verify_frozen(frozen,relocated,raw)
            (raw/'train.gz').write_bytes(b'changed')
            with self.assertRaises(ValueError): p.verify_frozen(frozen,relocated,raw)

if __name__=='__main__': unittest.main()

