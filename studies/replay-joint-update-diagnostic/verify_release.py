#!/usr/bin/env python3
"""Verify packaged bytes and saved scalar recount, never scientific inference."""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys

ROOT = Path(__file__).resolve().parent

def digest(data):
    return hashlib.sha256(data).hexdigest()

def git_blob(data):
    return hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()

def require(condition, message):
    if not condition:
        raise ValueError(message)

def read_json(relative):
    return json.loads((ROOT / relative).read_text())

def normalized_ast(text, kind):
    tree = ast.parse(text)
    kept = []
    for node in tree.body:
        if (kind == 'runner' and not kept and isinstance(node, ast.Expr)
                and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)):
            continue
        if isinstance(node, ast.Assign):
            names = {'HISTORY', 'DATA', 'SNAPSHOT'} if kind == 'runner' else {'ROOT', 'DEST'}
            if any(isinstance(t, ast.Name) and t.id in names for t in node.targets):
                continue
            if kind == 'recount' and any(
                    isinstance(t, ast.Subscript) and isinstance(t.value, ast.Name)
                    and t.value.id == 'report' and isinstance(t.slice, ast.Constant)
                    and t.slice.value == 'input_output_bindings' for t in node.targets):
                continue
        kept.append(node)
    tree.body = kept
    return ast.dump(tree, include_attributes=False)

def numeric_constants(text):
    values = [node.value for node in ast.walk(ast.parse(text))
              if isinstance(node, ast.Constant) and isinstance(node.value, (int, float, complex))
              and not isinstance(node.value, bool)]
    return digest(repr(values).encode())

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recount', action='store_true', help='Recount saved scalar JSON using NumPy; no model/data imports.')
    args = parser.parse_args()
    inventory = read_json('PUBLIC-FILES.json')
    require(inventory['format'] == 'public-file-allowlist-v1', 'unknown allowlist format')
    files = inventory['files']
    require('PUBLIC-FILES.json' not in files, 'allowlist must exclude its own bytes')
    actual = set()
    for path in ROOT.rglob('*'):
        require(not path.is_symlink(), 'symlinks are not allowed')
        if path.is_file() and '__pycache__' not in path.relative_to(ROOT).parts:
            actual.add(path.relative_to(ROOT).as_posix())
    require(actual == set(files) | {'PUBLIC-FILES.json'}, 'public file set does not match allowlist')
    for relative, expected in files.items():
        path = PurePosixPath(relative)
        require(not path.is_absolute() and '..' not in path.parts, 'unsafe allowlist path')
        data = (ROOT / relative).read_bytes()
        require(len(data) == expected['bytes'] and digest(data) == expected['sha256']
                and git_blob(data) == expected['git_blob_sha1'], 'allowlist mismatch: '+relative)

    result_dir = ROOT / 'results/diagnostic-v2'
    states = [f'seed-{seed}-block-{block}.json' for seed in range(2001, 2005) for block in range(4)]
    require({path.name for path in result_dir.iterdir()} == set(states) | {'summary.json', 'COMPLETE.json'},
            'unexpected retained-result file inventory')
    complete = read_json('results/diagnostic-v2/COMPLETE.json')
    require(complete['states'] == 16 and complete['seed_clusters'] == 4, 'wrong original design counts')
    require(set(complete['files']) == set(states) | {'summary.json', 'run-input-freeze.json'}, 'wrong original completion inventory')
    for name in states + ['summary.json']:
        data = (result_dir / name).read_bytes()
        expected = complete['files'][name]
        require(digest(data) == expected['sha256'] and len(data) == expected['bytes'], 'original result mismatch: '+name)
    for seed in range(2001, 2005):
        for block in range(4):
            row = read_json(f'results/diagnostic-v2/seed-{seed}-block-{block}.json')
            require(row['seed'] == seed and row['block'] == block and row['eta'] == .3,
                    'saved state identity mismatch')

    freeze = read_json('provenance/source-input-freeze.projection.json')
    execution = read_json('provenance/execution-summary.projection.json')
    require(freeze['is_executed_original'] is False and execution['is_original_receipt'] is False, 'projections mislabeled')
    require(freeze['original_sha256'] == complete['files']['run-input-freeze.json']['sha256'], 'original freeze commitment mismatch')
    require(freeze['original_bytes'] == complete['files']['run-input-freeze.json']['bytes'], 'original freeze size mismatch')
    require(len(freeze['original_file_bindings_role_relative']) == 55, 'wrong original input binding count')
    require(complete['artifact_bytes_including_complete'] == execution['original_completion_inclusive_bytes'] == 1060107,
            'original byte accounting mismatch')
    require(execution['scientific_attempts'] == 1 and execution['observed_external_exit'] == 0, 'unexpected recorded execution')
    saved = read_json('saved-analysis.json')
    bindings = saved['input_output_bindings']
    require(bindings['freeze_sha256'] == freeze['original_sha256'], 'saved freeze binding mismatch')
    require(bindings['observed_external_exit'] == execution['observed_external_exit'], 'saved exit commitment mismatch')
    for name, key in [('summary.json', 'summary_sha256'), ('COMPLETE.json', 'complete_sha256')]:
        require(digest((result_dir / name).read_bytes()) == bindings[key], 'saved result binding mismatch')
    require(set(bindings['source_rows_sha256']) == set(states), 'saved row binding inventory mismatch')
    for name in states:
        require(digest((result_dir / name).read_bytes()) == bindings['source_rows_sha256'][name], 'saved row digest mismatch')

    transformations = read_json('transformations.json')
    for item in transformations['files']:
        relative = item['packaged_path']
        require(files[relative] == item['packaged'], 'transformation identity mismatch: '+relative)
        if item['transformation'] == 'byte-identical':
            require(item['original']['sha256'] == item['packaged']['sha256'], 'incorrect exact-original claim')
    for kind, relative in [('runner', 'source/jointaudit/runner.py'), ('recount', 'recount_saved.py')]:
        text = (ROOT / relative).read_text()
        proof = transformations['source_equivalence'][kind]
        require(digest(normalized_ast(text, kind).encode()) == proof['normalized_ast_sha256'], 'normalized AST mismatch')
        require(numeric_constants(text) == proof['numeric_constants_sha256'], 'numeric AST constants mismatch')

    if args.recount:
        env = os.environ.copy()
        env.update({'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'PYTHONDONTWRITEBYTECODE': '1'})
        result = subprocess.run([sys.executable, '-B', str(ROOT / 'recount_saved.py')],
                                cwd=ROOT, env=env, capture_output=True, check=True)
        require(not result.stderr, 'unexpected recount stderr')
        require(result.stdout == (ROOT / 'saved-analysis.json').read_bytes(), 'recount is not byte-identical')
    print(json.dumps({'status': 'retained-public-evidence-verified', 'public_files_including_allowlist': len(files) + 1,
                      'exact_scientific_state_files': 16, 'original_complete_bound_files_verified': 17,
                      'original_freeze_commitment_only': True, 'full_original_bundle_verified': False,
                      'scientific_rerun_or_dataset_checkpoint_inference': False, 'saved_recount_byte_identical': True if args.recount else None},
                     indent=2, sort_keys=True))

if __name__ == '__main__':
    main()
