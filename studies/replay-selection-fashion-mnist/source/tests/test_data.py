"""Synthetic-only data integrity / leakage tests; never predict official test."""
import gzip
import hashlib
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

try:
    from replaystudy import data
except ImportError:
    data = None


class DataTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(data, "dataset preparation implementation must exist")

    def test_preprocess_exact_block_mean_float32(self):
        image = np.zeros((1, 28, 28), np.uint8)
        image[0, :2, :2] = [[0, 1], [2, 3]]
        transformed = data.preprocess_images(image)
        self.assertEqual(transformed.shape, (1, 196))
        self.assertEqual(transformed.dtype, np.dtype('float32'))
        self.assertEqual(transformed[0, 0], np.float32(1.5) / np.float32(255))
        with self.assertRaises(ValueError):
            data.preprocess_images(image.astype(np.float32))

    def test_idx_dimensions_count_and_length_are_verified(self):
        good = struct.pack('>IIII', 2051, 2, 28, 28) + bytes(2 * 784)
        self.assertEqual(data.decode_idx(good, images=True, expected_count=2).shape, (2, 28, 28))
        for bad in [good[:-1], good + b'0', struct.pack('>IIII', 2051, 2, 14, 56) + bytes(2 * 784)]:
            with self.assertRaises(ValueError):
                data.decode_idx(bad, images=True, expected_count=2)
        with self.assertRaises(ValueError):
            data.decode_idx(struct.pack('>II', 2049, 2) + bytes([0, 10]), images=False, expected_count=2)

    def test_initial_split_matches_single_pcg64_class_loop(self):
        y = np.repeat(np.arange(3, dtype=np.uint8), 10)
        a = data.make_split_indices(y, train_per_class=3, val_per_class=2, seed=73102)
        rng = np.random.Generator(np.random.PCG64(73102))
        expected = [rng.permutation(np.flatnonzero(y == c)) for c in range(3)]
        np.testing.assert_array_equal(a['train_ids'], np.concatenate([e[:3] for e in expected]))
        np.testing.assert_array_equal(a['val_ids'], np.concatenate([e[3:5] for e in expected]))
        self.assertFalse(a['amended'])

    def test_feature_grouped_amendment_prevents_cross_split_leak(self):
        y = np.repeat(np.arange(2, dtype=np.uint8), 20)
        initial = data.make_split_indices(y, train_per_class=3, val_per_class=2)
        hashes = np.array([f'{i:064x}'.encode() for i in range(len(y))], dtype='S64')
        hashes[initial['val_ids'][0]] = hashes[initial['train_ids'][0]]
        # Also exercise an exact feature group spanning contradictory labels.
        hashes[initial['val_ids'][-1]] = hashes[initial['train_ids'][0]]
        a = data.make_split_indices(y, hashes, train_per_class=3, val_per_class=2)
        b = data.make_split_indices(y, hashes, train_per_class=3, val_per_class=2)
        self.assertTrue(a['amended'])
        self.assertFalse(set(hashes[a['train_ids']]) & set(hashes[a['val_ids']]))
        np.testing.assert_array_equal(a['val_ids'], b['val_ids'])
        np.testing.assert_array_equal(np.bincount(y[a['train_ids']]), [3, 3])
        np.testing.assert_array_equal(np.bincount(y[a['val_ids']]), [2, 2])

    def test_grouped_split_cannot_silently_relax_counts(self):
        y = np.repeat(np.arange(2, dtype=np.uint8), 10)
        hashes = np.array([b'a' * 64] * len(y), dtype='S64')
        with self.assertRaises(data.DataGateError):
            data.make_split_indices(y, hashes, train_per_class=3, val_per_class=2)

    def test_test_exclusion_is_raw_or_transformed_label_independent(self):
        raw_ref = np.array([b'a', b'b'])
        transformed_ref = np.array([b'c', b'd'])
        raw_test = np.array([b'a', b'x', b'z', b'q'])
        transformed_test = np.array([b'u', b'd', b'v', b'w'])
        audit = data.audit_cross_split_duplicates(raw_ref, transformed_ref, raw_test, transformed_test)
        np.testing.assert_array_equal(audit['keep_ids'], [2, 3])
        self.assertEqual(audit['raw_match_items'], 1)
        self.assertEqual(audit['transformed_match_items'], 1)
        self.assertEqual(audit['excluded_items'], 2)
        self.assertNotIn('labels', audit)

    def test_transform_collision_detected_despite_different_raw_pixels(self):
        a = np.zeros((2, 28, 28), np.uint8)
        a[0, 0, 0] = 4
        a[1, 0, 1] = 4
        raw = data.feature_hashes(a)
        transformed = data.feature_hashes(data.preprocess_images(a))
        self.assertNotEqual(raw[0], raw[1])
        self.assertEqual(transformed[0], transformed[1])

    def test_retained_class_floor_is_gate_not_filter(self):
        y = np.repeat(np.arange(10), 501)
        counts = data.check_retained_class_counts(y, np.arange(len(y)), minimum=500)
        self.assertEqual(counts, [501] * 10)
        with self.assertRaises(data.DataGateError):
            data.check_retained_class_counts(y, np.arange(2, len(y)), minimum=500)

    def test_outputs_refuse_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'immutable.json'
            data.write_json_once(p, {'value': 1})
            original = p.read_bytes()
            with self.assertRaises(FileExistsError):
                data.write_json_once(p, {'value': 2})
            self.assertEqual(p.read_bytes(), original)

    def test_hash_validation_rejects_altered_bytes(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'fixture.gz'
            p.write_bytes(gzip.compress(b'example'))
            actual = data.file_hashes(p)
            self.assertEqual(actual['sha256'], hashlib.sha256(p.read_bytes()).hexdigest())
            self.assertEqual(actual['md5'], hashlib.md5(p.read_bytes()).hexdigest())
            with self.assertRaises(data.DataIntegrityError):
                data.verify_asset(p, expected_md5='0' * 32)

    def test_test_scoring_loader_fails_closed_before_reading_assets(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(data.DataGateError):
                data.load_test_for_scoring(Path(d), Path(d), lock_path=None, verify_lock=None)


    def test_synthetic_preparation_roundtrip_and_index_tamper_detection(self):
        # Real transform/audit/serialization against synthetic source arrays;
        # only network/source acquisition is substituted.
        rng = np.random.Generator(np.random.PCG64(44))
        assets = {
            'train_images': rng.integers(0, 256, (15000, 28, 28), dtype=np.uint8),
            'train_labels': np.repeat(np.arange(10, dtype=np.uint8), 1500),
            'test_images': rng.integers(0, 256, (5000, 28, 28), dtype=np.uint8),
            'test_labels': np.repeat(np.arange(10, dtype=np.uint8), 500),
        }
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            out = root / 'prepared-v1'
            source_manifest, pins = self._source_fixture(root)
            with patch.object(data, 'ASSETS', pins), patch.object(
                    data, 'download_verified_assets', return_value=source_manifest['assets']), patch.object(
                    data, 'load_verified_assets', return_value=assets):
                manifest = data.prepare_dataset(root, out)
                self.assertTrue(manifest['class_count_gate_passed'])
                self.assertFalse(manifest['test_predictions_performed'])
                self.assertEqual(manifest['retained_test_class_counts'], [500] * 10)
                self.assertIn('unresolved', manifest['prior_test_use_attestation'])
                loaded = data.load_development(root, out)
                self.assertEqual(set(loaded), {'x_train','y_train','train_ids','x_val','y_val','val_ids'})
                self.assertEqual(loaded['x_train'].shape, (10000, 196))
                self.assertEqual(loaded['x_val'].shape, (5000, 196))
                self.assertFalse(set(loaded['train_ids']) & set(loaded['val_ids']))
                with self.assertRaises(FileExistsError):
                    data.prepare_dataset(root, out)
                index_path = out / 'train_ids.npy'
                content = bytearray(index_path.read_bytes())
                content[-1] ^= 1
                index_path.write_bytes(content)
                with self.assertRaises(data.DataIntegrityError):
                    data.load_development(root, out)

    def test_download_refuses_unpinned_source_before_network(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(data.DataIntegrityError):
                data._download_once('https://example.org/train-images.gz', Path(d) / 'asset')

    def test_lock_boolean_or_wrong_manifest_hash_is_not_scoring_evidence(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            data.write_json_once(root / 'manifest.json', {'fixture': True})
            for evidence in (True, {}, {'complete': True, 'no_prior_test_selection': True,
                                        'data_manifest_sha256': '0' * 64}):
                with self.subTest(evidence=evidence):
                    with self.assertRaises(data.DataGateError):
                        data.load_test_for_scoring(root, root, lock_path=root / 'lock.json',
                                                   verify_lock=lambda _: evidence)

    def test_code_root_dataset_storage_is_rejected(self):
        with self.assertRaises(data.DataGateError):
            data.prepare_dataset(Path(data.__file__).resolve().parents[1] / 'dataset')



    def _source_fixture(self, root):
        raw = root / 'raw'
        raw.mkdir()
        entries, pins = {}, {}
        for i, (name, (_, count, images)) in enumerate(data.ASSETS.items()):
            payload = gzip.compress(('synthetic pinned source ' + str(i)).encode())
            (raw / name).write_bytes(payload)
            md5 = hashlib.md5(payload).hexdigest()
            pins[name] = (md5, count, images)
            entries[name] = data.file_hashes(raw / name) | {
                'url': data.BASE_URL + 'data/fashion/' + name, 'published_md5': md5}
        (raw / 'LICENSE').write_text('The MIT License (MIT) Copyright 2017 Zalando SE')
        entries['LICENSE'] = data.file_hashes(raw / 'LICENSE') | {
            'url': data.BASE_URL + 'LICENSE', 'license': 'MIT'}
        return {'assets': entries}, pins

    def test_all_raw_assets_match_manifest_sha256_size_and_published_md5(self):
        self.assertTrue(hasattr(data, 'verify_source_manifest'), 'source SHA256 verifier must exist')
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            manifest, pins = self._source_fixture(root)
            with patch.object(data, 'ASSETS', pins):
                data.verify_source_manifest(root, manifest)
                for name in manifest['assets']:
                    with self.subTest(name=name, field='sha256'):
                        altered = json.loads(json.dumps(manifest))
                        altered['assets'][name]['sha256'] = '0' * 64
                        with self.assertRaises(data.DataIntegrityError):
                            data.verify_source_manifest(root, altered)
                    with self.subTest(name=name, field='bytes'):
                        altered = json.loads(json.dumps(manifest))
                        altered['assets'][name]['bytes'] += 1
                        with self.assertRaises(data.DataIntegrityError):
                            data.verify_source_manifest(root, altered)
                name = next(iter(pins))
                altered = json.loads(json.dumps(manifest))
                altered['assets'][name]['published_md5'] = '0' * 32
                with self.assertRaises(data.DataIntegrityError):
                    data.verify_source_manifest(root, altered)
                altered = json.loads(json.dumps(manifest))
                del altered['assets']['LICENSE']
                with self.assertRaises(data.DataIntegrityError):
                    data.verify_source_manifest(root, altered)

    def test_raw_file_or_license_tamper_rejected(self):
        self.assertTrue(hasattr(data, 'verify_source_manifest'), 'source SHA256 verifier must exist')
        for target in list(data.ASSETS) + ['LICENSE']:
            with self.subTest(target=target), tempfile.TemporaryDirectory() as d:
                root = Path(d)
                manifest, pins = self._source_fixture(root)
                p = root / 'raw' / target
                payload = bytearray(p.read_bytes())
                payload[-1] ^= 1  # Same byte length, so SHA/MD5 must catch it.
                p.write_bytes(payload)
                with patch.object(data, 'ASSETS', pins):
                    with self.assertRaises(data.DataIntegrityError):
                        data.verify_source_manifest(root, manifest)

    def test_both_loaders_check_all_raw_sources_before_returning_features(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            manifest, pins = self._source_fixture(root)
            manifest['class_count_gate_passed'] = True
            data.write_json_once(root / 'manifest.json', manifest)
            lock = {'complete': True, 'no_prior_test_selection': True,
                    'data_manifest_sha256': data.file_hashes(root / 'manifest.json')['sha256']}
            # A corrupted *training* asset must also block the test unlock.
            name = 'train-images-idx3-ubyte.gz'
            (root / 'raw' / name).write_bytes(b'corrupt')
            with patch.object(data, 'ASSETS', pins), patch.object(
                    data, '_verified_preparation', return_value=(manifest, {})), patch.object(
                    data, 'load_verified_assets', side_effect=AssertionError('Reached development feature loading')), patch.object(
                    data, '_read_asset', side_effect=AssertionError('Reached test feature loading')):
                with self.assertRaises(data.DataIntegrityError):
                    data.load_development(root, root)
                with self.assertRaises(data.DataIntegrityError):
                    data.load_test_for_scoring(root, root, lock_path=root / 'lock.json', verify_lock=lambda _: lock)


if __name__ == '__main__':
    unittest.main()
