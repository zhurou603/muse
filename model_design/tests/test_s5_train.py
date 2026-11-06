from __future__ import annotations

import unittest

import numpy as np

from muse_encoder.s5.train import registered_structure_failure_rows


class _Store:
    def __init__(self):
        self.records = np.asarray([
            (b"a", 4), (b"b", 9), (b"c", 17),
        ], dtype=[("id", "S64"), ("aux_row", "<i8")])


class _Dataset:
    kind = "aux_molecule"
    use_molecule3d = True
    name = "toy"

    def __init__(self, failures):
        self._structure_failures = failures
        self._store = _Store()

    def _open_structure(self):
        return self._store


class RegisteredStructureFailureRowsTest(unittest.TestCase):
    def test_maps_audited_ids_to_aux_rows(self):
        dataset = _Dataset({"c", "a"})
        self.assertEqual(registered_structure_failure_rows(dataset), {4, 17})

    def test_unknown_failure_id_is_rejected(self):
        dataset = _Dataset({"missing"})
        with self.assertRaisesRegex(ValueError, "未登记 ID"):
            registered_structure_failure_rows(dataset)


if __name__ == "__main__":
    unittest.main()
