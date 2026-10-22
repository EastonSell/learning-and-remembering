"""Offline safety/integrity tests; synthetic arrays, never UCI training."""
import io
import tempfile
import unittest
import zipfile
import warnings
from pathlib import Path
import numpy as np
from functional_circuits.replay_data import (parse_rows, split_training, audit_duplicates,
                                            inspect_archive, write_report, validate, CLASS_COUNTS)


class ReplayDataTests(unittest.TestCase):
    def test_parser_rejects_bad_values_before_cast(self):
        good = ','.join(['0'] * 64 + ['9']) + '\n'
        x, y = parse_rows(good.encode(), 1)
        self.assertEqual(x.shape, (1, 64))
        self.assertEqual(y.tolist(), [9])
        for bad in [good.replace('0', '0.5', 1), good.replace('0', 'nan', 1),
                    good.replace('0', '17', 1), good.replace('9', '10'),
                    good.replace('0', '999999999999999999999', 1),
                    good + good, good.replace(',', '', 1)]:
            with self.subTest(bad=bad[:20]), self.assertRaises(ValueError):
                parse_rows(bad.encode(), 1)

    def test_split_is_deterministic_disjoint_and_stratified(self):
        y = np.repeat(np.arange(10), 11)
        a, b = split_training(y)
        c, d = split_training(y)
        np.testing.assert_array_equal(a, c)
        np.testing.assert_array_equal(b, d)
        self.assertFalse(set(a) & set(b))
        self.assertEqual(set(a) | set(b), set(range(110)))
        self.assertEqual(np.bincount(y[b]).tolist(), [2] * 10)
        with self.assertRaises(ValueError):
            split_training(np.array([0, 0]))

    def test_duplicates_block_cross_split_and_conflicts_not_training_repeats(self):
        x = np.zeros((3, 64), dtype=np.int16)
        report = audit_duplicates({'train': (x, np.array([0, 0, 0]), [0, 1, 2])})
        self.assertTrue(report['passed'])
        self.assertEqual(report['duplicate_groups'][0]['rows'], 3)
        report = audit_duplicates({'train': (x[:1], [0], [0]),
                                   'validation': (x[1:2], [0], [1])})
        self.assertFalse(report['passed'])
        self.assertEqual(report['cross_split_groups'], 1)
        report = audit_duplicates({'train': (x[:2], [0, 1], [0, 1])})
        self.assertEqual(report['conflicting_label_groups'], 1)
        self.assertFalse(report['passed'])

    def test_zip_rejects_unsafe_duplicate_and_oversized_members(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'bad.zip'
            for name in ['../outside', '/outside', 'C:/outside', 'foo\\bar']:
                with zipfile.ZipFile(path, 'w') as z:
                    z.writestr(name, b'x')
                with self.assertRaises(ValueError):
                    inspect_archive(path)
            with zipfile.ZipFile(path, 'w') as z:
                z.writestr('small', b'x')
            with self.assertRaises(ValueError):
                inspect_archive(path, max_compressed=1)
            with self.assertRaises(ValueError):
                inspect_archive(path, max_expanded=0)

    def test_full_synthetic_archive_and_duplicate_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'data.zip'
            offset = 0
            with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as z:
                for name, counts in CLASS_COUNTS.items():
                    labels = np.repeat(np.arange(10), counts)
                    rows = []
                    for i, label in enumerate(labels):
                        value = i + offset
                        features = [(value // 17**j) % 17 for j in range(4)] + [0]*60
                        rows.append(','.join(map(str, features + [int(label)])))
                    z.writestr(name, '\n'.join(rows) + '\n')
                    offset += len(labels)
            report = validate(path)
            self.assertEqual(report['status'], 'passed')
            self.assertFalse(report['training_performed'])
            self.assertEqual(report['split_counts'], dict(train=3062, validation=761, test=1797))
            self.assertEqual(report, validate(path))
            with zipfile.ZipFile(path, 'w') as z, warnings.catch_warnings():
                warnings.simplefilter('ignore', UserWarning)
                z.writestr('duplicate', b'one')
                z.writestr('duplicate', b'two')
            with self.assertRaises(ValueError):
                inspect_archive(path)

    def test_report_never_overwrites(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'report.json'
            write_report(p, {'passed': True})
            with self.assertRaises(FileExistsError):
                write_report(p, {'passed': False})
            self.assertIn('true', p.read_text())


if __name__ == '__main__':
    unittest.main()
