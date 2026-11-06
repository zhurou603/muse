from __future__ import annotations

import csv
import gzip
import io
import json
import pickle
import sys
import tarfile
import tempfile
import unittest
import zlib
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DESIGN = PROJECT_ROOT / "model_design"
for path in (str(PROJECT_ROOT), str(MODEL_DESIGN)):
    if path not in sys.path:
        sys.path.insert(0, path)

from muse import tasks as T
from muse_encoder.config import MuseConfig
from muse_encoder.featurize import D_RESIDUE_CHIRAL_CODE, RecordFeaturizer
from muse_encoder.model import MuseEncoder
from muse_encoder.pretrain import dashboard as D
from muse_encoder.tokenizers import load_tokenizers


FAST_PROBE = D.ProbeConfig(seeds=(0,), max_epochs=2, patience=1, batch_size=16)


def small_config() -> MuseConfig:
    tokenizers = load_tokenizers()
    return MuseConfig(
        smiles_vocab_size=len(tokenizers["smiles"]),
        psmiles_vocab_size=len(tokenizers["psmiles"]),
        residue_vocab_size=len(tokenizers["residue"]),
        d_model=32,
        n_layers=2,
        n_heads=4,
        d_ffn=48,
        dropout=0.0,
        max_seq_len=32,
        num_experts=2,
        mole_every=1,
        embed_dim=24,
        contrastive_dim=8,
        descriptor_dim=8,
    )


def make_checkpoint(root: Path) -> tuple[Path, MuseConfig, MuseEncoder, str]:
    cfg = small_config()
    torch.manual_seed(17)
    model = MuseEncoder(cfg)
    path = root / "toy.pt"
    torch.save({
        "config": cfg.to_dict(),
        "model": model.state_dict(),
        "state": {"step": 0, "micro_step": 0, "samples": 0},
    }, path)
    return path, cfg, model, D.sha256_file(path)


class ProbeAndCheckpointTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_probe_regression_binary_multiclass_and_budget(self):
        rng = np.random.default_rng(5)
        x = rng.normal(size=(36, 12)).astype(np.float32)
        xv = rng.normal(size=(8, 12)).astype(np.float32)
        y = 2.0 * x[:, 0] - x[:, 1]
        pred, meta = D.fit_probe(x, y, xv, "regression", config=FAST_PROBE)
        self.assertEqual(pred.shape, (8,))
        self.assertEqual(meta["hidden"], 48)
        self.assertFalse(meta["budget_adapted"])
        self.assertEqual(meta["params_per_head"], 12 * 48 + 48 + 48 + 1)

        scorer = D.official_scorer()
        positive = scorer.encode_class_label(1)
        ybin = [i % 2 for i in range(36)]
        pred, meta = D.fit_probe(
            x, ybin, xv, "binary_classification", positive_label=positive,
            config=FAST_PROBE,
        )
        self.assertEqual(pred.shape, (8,))
        self.assertTrue(np.all((pred >= 0) & (pred <= 1)))

        classes = tuple(scorer.encode_class_label(i) for i in range(3))
        ymulti = [i % 3 for i in range(36)]
        pred, meta = D.fit_probe(
            x, ymulti, xv, "multiclass_classification", class_labels=classes,
            config=FAST_PROBE,
        )
        self.assertEqual(pred.shape, (8, 3))
        np.testing.assert_allclose(pred.sum(axis=1), 1.0, rtol=1e-5, atol=1e-5)
        hidden, adapted, params = D.probe_hidden_dim(1024, 1195)
        self.assertEqual((hidden, adapted), (44, True))
        self.assertLessEqual(params, 100_000)

    def test_checkpoint_load_sha_and_public_weight_rejection(self):
        path, cfg, model, sha = make_checkpoint(self.root)
        loaded, loaded_cfg, info = D.load_checkpoint(path)
        self.assertEqual(info["sha256"], sha)
        self.assertEqual(loaded_cfg.embed_dim, cfg.embed_dim)
        self.assertEqual(info["state"]["step"], 0)
        for left, right in zip(loaded.state_dict().values(), model.state_dict().values()):
            self.assertTrue(torch.equal(left, right))
            break
        with self.assertRaisesRegex(ValueError, "公开模型目录"):
            D.load_checkpoint(D.PUBLIC_MODEL_ROOT / "public-checkpoint.pt")

    def test_shadow_aux_regression_interface(self):
        _path, cfg, model, _sha = make_checkpoint(self.root)
        records = []
        for index in range(30):
            records.append({
                "id": f"aux-{index}",
                "sample_type": "protein",
                "inputs": {"sequence": "ACDE"[:1 + index % 4]},
                "external_inputs": [],
                "labels": {"heldout": [{"value": float(index % 7), "task_type": "regression"}]},
            })
        result = D.evaluate_shadow_records(
            model, RecordFeaturizer(cfg), cfg, "heldout", records, 8, "cpu", FAST_PROBE
        )
        self.assertEqual(result["status"], "ok")
        self.assertGreater(result["n_train"], 0)
        self.assertGreater(result["n_validation"], 0)
        self.assertIn("normalized_score", result)

        constructs = [dict(record, sample_type="protein_construct") for record in records]
        normalized = [D.normalize_record(record) for record in constructs]
        normalized_result = D.evaluate_shadow_records(
            model, RecordFeaturizer(cfg), cfg, "heldout", normalized, 8, "cpu", FAST_PROBE
        )
        self.assertEqual(normalized_result["n_train"], result["n_train"])
        self.assertEqual(normalized_result["n_validation"], result["n_validation"])
        self.assertAlmostEqual(normalized_result["normalized_score"], result["normalized_score"])

        namespaces = []

        def custom_encoder(encoder, featurizer, config, rows, batch_size, token_level, device,
                           namespace):
            namespaces.append(namespace)
            return D.encode_records(encoder, featurizer, config, rows, batch_size, token_level, device)

        injected = D.evaluate_shadow_records(
            model, RecordFeaturizer(cfg), cfg, "heldout", normalized, 8, "cpu", FAST_PROBE,
            record_encoder=custom_encoder, namespace="test::heldout")
        self.assertEqual(injected["n_train"], result["n_train"])
        self.assertEqual(injected["n_validation"], result["n_validation"])
        self.assertEqual(namespaces, ["test::heldout/train", "test::heldout/validation"])

        capped = D.evaluate_shadow_records(
            model, RecordFeaturizer(cfg), cfg, "heldout", normalized, 8, "cpu", FAST_PROBE,
            train_cap=8)
        self.assertEqual(capped["train_cap"], 8)
        self.assertLessEqual(capped["n_train"], 8)
        self.assertEqual(capped["n_train_available"], result["n_train"])
        self.assertEqual(capped["n_validation"], result["n_validation"])

    def test_shadow_dipole_uses_validated_norm(self):
        observation = {"task_type": "vector_regression", "value": [3.0, 4.0, 0.0, 5.0]}
        self.assertEqual(D._shadow_regression_value("gfn2_dipole", observation), 5.0)
        observation["value"][-1] = 6.0
        with self.assertRaisesRegex(ValueError, "模长不一致"):
            D._shadow_regression_value("gfn2_dipole", observation)
        self.assertIsNone(D._shadow_regression_value("other_vector", observation))

        from types import SimpleNamespace
        from unittest.mock import patch
        payload = {"id": "dipole", "sample_type": "conformer",
                   "labels": {"gfn2_dipole": [
                       {"task_type": "vector_regression", "value": [3.0, 4.0, 0.0, 5.0]}],
                       "unrelated": [{"task_type": "atom_regression", "value": [1.0, 2.0]}]}}
        with patch("muse_encoder.pretrain.aux.iter_source",
                   return_value=iter([SimpleNamespace(payload=payload)])):
            selected = D._collect_shadow_records("qmugs", "gfn2_dipole", None)
        self.assertEqual(list(selected[0]["labels"]), ["gfn2_dipole"])
        self.assertIn("unrelated", payload["labels"])

    def test_token_bounded_ranges(self):
        from muse_encoder.inference import _token_bounded_ranges

        cfg = small_config()
        self.assertEqual(list(_token_bounded_ranges([32] * 20, cfg)),
                         [(0, 16), (16, 20)])
        self.assertEqual(list(_token_bounded_ranges([20, 32, 20, 1], cfg, max_batch_size=2)),
                         [(0, 2), (2, 4)])


class InputNormalizationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.cfg = small_config()
        self.featurizer = RecordFeaturizer(self.cfg)

    def tearDown(self):
        self.tmp.cleanup()

    def test_dbaasp_context_and_lowercase_are_explicit(self):
        raw = {
            "id": "pep-1",
            "sample_type": "peptide_measurement",
            "inputs": {
                "sequence": "ACdX",
                "context": {"target": "E. coli", "measurement_group": "MIC", "medium": "MHB"},
            },
            "external_inputs": [],
            "labels": {},
        }
        normalized = D.normalize_record(raw)
        prepared = D._prepare_featurized(self.featurizer, normalized)
        self.assertEqual(normalized["_peptide_stats"]["lowercase_d_residues"], 1)
        self.assertNotEqual(normalized["inputs"]["sequence"], normalized["inputs"]["sequence"].upper())
        self.assertGreater(np.abs(prepared.scalar_cond).sum(), 0)
        self.assertEqual(prepared.atom_features[3, 1], D_RESIDUE_CHIRAL_CODE)
        unknown = dict(raw)
        unknown["inputs"] = dict(raw["inputs"], sequence="ACx")
        unknown_normalized = D.normalize_record(unknown)
        unknown_prepared = D._prepare_featurized(self.featurizer, unknown_normalized)
        self.assertEqual(unknown_normalized["inputs"]["sequence"], "ACX")
        self.assertEqual(unknown_normalized["_peptide_stats"]["ambiguous_residues"], 1)
        self.assertNotEqual(unknown_prepared.atom_features[3, 1], D_RESIDUE_CHIRAL_CODE)
        self.assertEqual(unknown_prepared.token_ids[3], load_tokenizers()["residue"].vocab["X"])
        bad = dict(raw)
        bad["inputs"] = {"sequence": "R2(R2)KW3", "context": {}}
        with self.assertRaisesRegex(ValueError, "修饰记号"):
            D.normalize_record(bad)

    def test_polymer_electrolyte_variants(self):
        pe1 = {
            "id": "pe1", "sample_type": "polymer_electrolyte",
            "inputs": {"electrolyte_string": "*CCO*$x|CO$...$|1.5$2.5$mole_ratio"},
            "external_inputs": [], "labels": {},
        }
        out = D.normalize_record(pe1)
        self.assertEqual(out["sample_type"], "polymer")
        self.assertIn("*CCO*.CO", out["inputs"]["psmiles"])
        self.assertEqual(out["_scalar_cond"].shape, (4,))
        pe2 = {
            "id": "pe2", "sample_type": "polymer_electrolyte",
            "inputs": {"context": {
                "SMILES descriptor 1": "*CCO*", "SMILES descriptor 2": "*C*",
                "Amion SMILES": "O=S=O", "log Li:functional group": 0.1,
                "Comonomer percentage": 50.0, "approxMW(kDa)": 10.0,
                "approxTg": -20.0, "chain_architecture": "S_2",
            }},
            "external_inputs": [], "labels": {},
        }
        out = D.normalize_record(pe2)
        self.assertEqual(out["sample_type"], "polymer")
        self.assertIn("chain_architecture", out["_normalization"])

    def test_pei_single_section_and_dummy_marker(self):
        raw = {
            "id": "pe-single", "sample_type": "polymer_electrolyte",
            "inputs": {"electrolyte_string": "*CCO*.C[^]$0.5$NAN_ratio$95"},
            "external_inputs": [], "labels": {},
        }
        out = D.normalize_record(raw)
        self.assertEqual(out["inputs"]["psmiles"], "*CCO*.C*")
        self.assertEqual(out["_scalar_cond"][1:].tolist(), [0.0, 0.0, 0.0])
        self.assertIn("dummy atoms", out["_normalization"])
        prepared = D._prepare_featurized(self.featurizer, out)
        self.assertGreater(prepared.n_real_tokens, 0)

    def test_sdf_error_isolated_to_bad_record(self):
        from rdkit import Chem

        block = Chem.MolToMolBlock(Chem.MolFromSmiles("CCO"))
        block = "good" + block[block.index("\n"):]
        (self.root / "mixed.sdf").write_text(
            block + "$$$$\nbad\nnot an sdf\n$$$$\n", encoding="utf-8")
        found, failed = D._read_sdf_selected(self.root / "mixed.sdf", {"good", "bad"})
        self.assertEqual(set(found), {"good"})
        self.assertIn("bad", failed)
        records = [
            {"id": key, "sample_type": "molecule", "inputs": {"smiles": "CCO"},
             "external_inputs": [{"role": "molecular_structure", "format": "sdf",
                                  "selector": {"record_id": key}, "source_path": "mixed.sdf"}],
             "labels": {}}
            for key in ("good", "bad")
        ]
        resolver = D.ExternalStructureResolver(self.root, self.root / "mixed-cache")
        resolver.prepare(records)
        self.assertEqual(D.normalize_record(records[0], resolver)["id"], "good")
        with self.assertRaisesRegex(ValueError, "SDF"):
            D.normalize_record(records[1], resolver)

    def test_official_graph_variant_preparation(self):
        from rdkit import Chem

        mol = Chem.AddHs(Chem.MolFromSmiles("C[S@@+](C)[O-]"))
        conformer = Chem.Conformer(mol.GetNumAtoms())
        for index in range(mol.GetNumAtoms()):
            conformer.SetAtomPosition(index, (0.25 * index, 0.5 * index, 0.75 * index))
        mol.AddConformer(conformer)
        raw = {"id": "variant", "sample_type": "conformer",
               "inputs": {"smiles": "CS(C)=O", "sdf_mol": mol}}
        prepared = D._prepare_featurized(self.featurizer, raw)
        self.assertIn("official conformer graph", prepared.meta["normalization"])
        self.assertIsNotNone(prepared.coords)
        self.assertIsNotNone(prepared.meta["sdf_mapping"])

    def test_registered_official_graph_repair(self):
        from rdkit import Chem
        from unittest.mock import patch
        from muse_encoder.s5.official_graph_repairs import make_repair_entry

        mol = Chem.MolFromSmiles("CCO")
        mol.SetProp("_Name", "registered")
        conformer = Chem.Conformer(mol.GetNumAtoms())
        for index in range(mol.GetNumAtoms()):
            conformer.SetAtomPosition(index, (float(index), 0.1 * index, 0.0))
        mol.AddConformer(conformer)
        block = Chem.MolToMolBlock(mol)
        source = self.root / "qm9.sdf"
        source.write_text(block, encoding="utf-8")
        raw = {
            "id": "registered", "sample_type": "molecule",
            "inputs": {"smiles": "not-a-molecule"},
            "external_inputs": [{"role": "molecular_structure", "format": "sdf",
                                  "selector": {"record_id": "registered"},
                                  "source_path": "qm9.sdf"}],
            "labels": {},
        }
        serialized = Chem.MolFromMolBlock(block, sanitize=False, removeHs=False)
        entry = make_repair_entry(
            raw, serialized, block, self.featurizer, "qm9.sdf", D.sha256_file(source),
            "strict", ["synthetic invalid auxiliary input"], artifact_root=self.root / "artifacts")
        manifest = self.root / "repairs.json"
        manifest.write_text(json.dumps({
            "schema": "muse_encoder.official_graph_repairs.v1", "source": "qm9",
            "source_path": "qm9.sdf", "source_sha256": D.sha256_file(source),
            "artifact_dir": "artifacts", "repairs": [entry],
        }), encoding="utf-8")
        with patch.dict(D._AUX_EXTERNAL_BASES, {"qm9": self.root}):
            D.configure_official_graph_repairs(manifest)
            try:
                normalized = D.normalize_record(raw)
                prepared = D._prepare_featurized(self.featurizer, normalized)
                self.assertEqual(prepared.n_real_tokens, 3)
                self.assertIsNotNone(prepared.coords)
                self.assertEqual(prepared.meta["official_graph_repair"]["record_id"], "registered")
                with patch("muse_encoder.pretrain.dashboard._featurize_record",
                           side_effect=AssertionError("registered graph must take precedence")):
                    direct = D._prepare_featurized(self.featurizer, normalized)
                self.assertEqual(direct.meta["official_graph_repair"]["record_id"], "registered")
                tampered = dict(raw, inputs={"smiles": "different-invalid"})
                with self.assertRaisesRegex(RuntimeError, "原始输入指纹"):
                    D.normalize_record(tampered)
                plain = {"id": "plain", "sample_type": "molecule",
                         "inputs": {"smiles": "CCO"}}
                ordinary = D._prepare_featurized(self.featurizer, D.normalize_record(plain))
                self.assertNotIn("official_graph_repair", ordinary.meta)
            finally:
                D.configure_official_graph_repairs(None)

    def test_registered_sequence_repairs_are_explicit_and_fingerprinted(self):
        from unittest.mock import patch

        records = [
            {"id": "sequence-molecule", "sample_type": "molecule",
             "inputs": {"smiles": "not-a-molecule"}, "external_inputs": []},
            {"id": "sequence-complex", "sample_type": "protein_ligand",
             "inputs": {"protein_sequence": "ACd", "ligand_smiles": "not-a-molecule"},
             "external_inputs": []},
            {"id": "sequence-peptide", "sample_type": "peptide_measurement",
             "inputs": {"sequence": "R2(R2)KW3", "context": {}}, "external_inputs": []},
            {"id": "missing-polymer", "sample_type": "polymer",
             "inputs": {"psmiles": "nan"}, "external_inputs": []},
        ]
        source = self.root / "inputs.jsonl"
        source.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
        representations = (
            "smiles_sequence", "smiles_sequence", "raw_peptide_sequence",
            "missing_polymer_sequence",
        )
        entries = [{
            "record_id": record["id"], "representation": representation,
            "input_fingerprint": D._input_repair_fingerprint(record),
            "source_jsonl": str(source), "source_jsonl_sha256": D.sha256_file(source),
            "original_errors": ["synthetic invalid input"],
        } for record, representation in zip(records, representations)]
        manifest = self.root / "input-repairs.json"
        manifest.write_text(json.dumps({
            "schema": "muse_encoder.input_repairs.v1", "contains_labels": False,
            "artifact_dir": "artifacts", "repairs": entries,
        }), encoding="utf-8")
        with patch.object(D, "DATA_ROOT", self.root):
            D.configure_official_graph_repairs(manifest)
            try:
                prepared = [D._prepare_featurized(
                    self.featurizer, D.normalize_record(record)) for record in records]
                self.assertTrue(all(record.meta["degraded"] for record in prepared))
                self.assertIsNone(prepared[0].coords)
                self.assertTrue(prepared[0].meta["sequence_frontend"])
                self.assertTrue(prepared[1].meta["ligand_sequence_frontend"])
                self.assertEqual(prepared[1].meta["protein_length"], 3)
                stats = prepared[2].meta["peptide_stats"]
                self.assertTrue(stats["uninterpreted_modification_notation"])
                self.assertEqual(prepared[2].meta["input_repair"]["record_id"], "sequence-peptide")
                self.assertEqual(prepared[3].n_real_tokens, 1)
                self.assertEqual(prepared[3].token_ids[1], load_tokenizers()["psmiles"].unk_id)
                self.assertEqual(prepared[3].meta["original_psmiles"], "nan")
                tampered = dict(records[0], inputs={"smiles": "different-invalid"})
                with self.assertRaisesRegex(RuntimeError, "原始输入指纹"):
                    D.normalize_record(tampered)
            finally:
                D.configure_official_graph_repairs(None)

    def test_charged_geom_repair_serializes_scalar_condition(self):
        from rdkit import Chem
        from unittest.mock import patch
        from muse_encoder.s5.official_graph_repairs import make_repair_entry

        mol = Chem.MolFromSmiles("[NH4+]")
        conformer = Chem.Conformer(mol.GetNumAtoms())
        conformer.SetAtomPosition(0, (0.0, 0.0, 0.0))
        mol.AddConformer(conformer)
        block = Chem.MolToMolBlock(mol)
        source = self.root / "geom.tar"
        source.write_bytes(b"synthetic-source")
        raw = {
            "id": "charged-geom", "sample_type": "conformer",
            "inputs": {"smiles": "[NH4+]", "parent_molecule_id": "parent", "charge": 1},
            "external_inputs": [{"role": "molecular_conformer", "format": "rdkit_pickle",
                                 "selector": {"archive_member": "a.pickle", "conformer_index": 0,
                                              "geom_id": 1}, "source_path": "geom.tar"}],
        }
        entry = make_repair_entry(
            raw, mol, block, self.featurizer, "geom.tar", D.sha256_file(source),
            "official_rdkit_conformer", ["synthetic graph mismatch"],
            artifact_root=self.root / "charged-artifacts", source_format="rdkit_pickle")
        manifest = self.root / "charged-repairs.json"
        manifest.write_text(json.dumps({
            "schema": "muse_encoder.input_repairs.v1", "contains_labels": False,
            "artifact_dir": "charged-artifacts", "repairs": [entry],
        }), encoding="utf-8")
        with patch.dict(D._AUX_EXTERNAL_BASES, {"geom_drugs": self.root}):
            D.configure_official_graph_repairs(manifest)
            try:
                prepared = D._prepare_featurized(self.featurizer, D.normalize_record(raw))
                self.assertAlmostEqual(float(prepared.scalar_cond[0]), 0.1, places=6)
                self.assertEqual(prepared.meta["official_graph_repair"]["source_format"],
                                 "rdkit_pickle")
            finally:
                D.configure_official_graph_repairs(None)

    def test_unsanitized_official_graph_preserves_hydrogen_count(self):
        from rdkit import Chem
        from muse_encoder.featurize import _NUM_HS

        mol = Chem.AddHs(Chem.MolFromSmiles("[NH4+]"))
        mol.GetAtomWithIdx(0).SetFormalCharge(0)
        conformer = Chem.Conformer(mol.GetNumAtoms())
        for index in range(mol.GetNumAtoms()):
            conformer.SetAtomPosition(index, (float(index), 0.1 * index, 0.0))
        mol.AddConformer(conformer)
        block = Chem.MolToMolBlock(mol)
        strict = Chem.MolFromMolBlock(block, sanitize=True, removeHs=False)
        self.assertIsNone(strict)
        unsanitized = Chem.MolFromMolBlock(block, sanitize=False, removeHs=False)
        prepared = self.featurizer.featurize_official_molecule(
            unsanitized, record_id="unsanitized", repair={"record_id": "unsanitized"})
        self.assertEqual(prepared.n_real_tokens, 1)
        self.assertEqual(prepared.atom_features[1, 4], _NUM_HS.index(4))
        self.assertEqual(prepared.coords.shape, (2, 3))

    def test_sdf_geom_cif_and_ase_lmdb_real_parsing(self):
        from rdkit import Chem

        mol = Chem.MolFromSmiles("CCO")
        conformer = Chem.Conformer(mol.GetNumAtoms())
        for index in range(mol.GetNumAtoms()):
            conformer.SetAtomPosition(index, (float(index), 0.1 * index, 0.0))
        mol.AddConformer(conformer)
        mol.SetProp("_Name", "sdf-1")
        sdf = self.root / "mols.sdf"
        with Chem.SDWriter(str(sdf)) as writer:
            writer.write(mol)
        record = {
            "id": "sdf-record", "sample_type": "molecule", "inputs": {"smiles": "CCO"},
            "external_inputs": [{"role": "molecular_structure", "format": "sdf",
                                  "selector": {"record_id": "sdf-1"},
                                  "source_path": "mols.sdf"}],
            "labels": {},
        }
        resolver = D.ExternalStructureResolver(self.root, self.root / "cache")
        resolver.prepare([record])
        out = D.normalize_record(record, resolver)
        self.assertIs(out["inputs"]["sdf_mol"].GetNumAtoms(), mol.GetNumAtoms())

        tar_path = self.root / "geom.tar"
        with tarfile.open(tar_path, "w") as archive:
            payload = pickle.dumps({"smiles": "CCO", "conformers": [{"geom_id": 1, "rd_mol": mol}]})
            member = tarfile.TarInfo("rdkit_folder/a.pickle")
            member.size = len(payload)
            archive.addfile(member, io.BytesIO(payload))
        geom = dict(record)
        geom.update({"id": "geom-record", "sample_type": "conformer",
                     "external_inputs": [{"role": "molecular_conformer", "format": "rdkit_pickle",
                                           "selector": {"archive_member": "rdkit_folder/a.pickle",
                                                         "conformer_index": 0, "geom_id": 1},
                                           "source_path": "geom.tar"}]})
        resolver.prepare([geom])
        out = D.normalize_record(geom, resolver)
        self.assertEqual(out["inputs"]["sdf_mol"].GetNumConformers(), 1)

        cif = self.root / "toy.cif"
        cif.write_text("""data_toy
_cell_length_a 5.0
_cell_length_b 5.0
_cell_length_c 5.0
_cell_angle_alpha 90.0
_cell_angle_beta 90.0
_cell_angle_gamma 90.0
_space_group_IT_number 1
loop_
_space_group_symop_operation_xyz
'x, y, z'
loop_
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Si 0.0 0.0 0.0
""", encoding="utf-8")
        periodic = {
            "id": "cif-record", "sample_type": "periodic_structure",
            "inputs": {"chemical_formula": "Si"},
            "external_inputs": [{"role": "periodic_structure", "format": "cif",
                                  "selector": {}, "source_path": "toy.cif"}],
            "labels": {},
        }
        out = D.normalize_record(periodic, resolver)
        self.assertEqual(out["sample_type"], "periodic_frame")
        self.assertEqual(out["inputs"]["elements"], ["Si"])
        self.assertEqual(out["inputs"]["spacegroup_id"], 1)

        import lmdb

        lmdb_path = self.root / "toy.aselmdb"
        env = lmdb.open(str(lmdb_path), subdir=False, map_size=4 * 1024 * 1024)
        payload = {
            "numbers": {"__ndarray__": [[2], "int64", [1, 6]]},
            "positions": {"__ndarray__": [[2, 3], "float64", [0, 0, 0, 1, 2, 3]]},
            "cell": {"__ndarray__": [[3, 3], "float64", [5, 0, 0, 0, 6, 0, 0, 0, 7]]},
            "pbc": {"__ndarray__": [[3], "bool", [True, True, True]]},
            "unique_id": "toy",
            "data": {"sid": "toy"},
        }
        with env.begin(write=True) as txn:
            txn.put(b"1", zlib.compress(json.dumps(payload).encode()))
        env.close()
        lmdb_tar = self.root / "lmdb.tar"
        with tarfile.open(lmdb_tar, "w") as archive:
            archive.add(lmdb_path, arcname="train/data0000.aselmdb", recursive=False)
        frame = {
            "id": "lmdb-record", "sample_type": "periodic_frame",
            "inputs": {"chemical_formula": "CH", "atom_count": 2},
            "external_inputs": [{"role": "periodic_structure", "format": "ase_lmdb",
                                  "selector": {"archive_member": "train/data0000.aselmdb",
                                                "lmdb_row_id": 1, "sid": "toy"},
                                  "source_path": "lmdb.tar"}],
            "labels": {},
        }
        out = D.normalize_record(frame, resolver)
        self.assertEqual(out["inputs"]["elements"], ["H", "C"])
        torch.manual_seed(11)
        encoded = D.encode_records(MuseEncoder(self.cfg), self.featurizer, self.cfg,
                                   [out], batch_size=1)
        self.assertEqual(encoded.embeddings.shape, (1, self.cfg.embed_dim))
        self.assertFalse(encoded.failed)

    def test_token_level_long_sequence_encoding(self):
        torch.manual_seed(3)
        model = MuseEncoder(self.cfg)
        records = [{
            "id": "long", "sample_type": "protein",
            "inputs": {"sequence": "ACDEFGHIKLMNPQRSTVWYACDE"},
            "external_inputs": [], "labels": {},
        }]
        encoded = D.encode_records(model, self.featurizer, self.cfg, records,
                                   batch_size=2, token_level=True)
        self.assertEqual(encoded.ids, ["long"])
        self.assertEqual(encoded.embeddings.shape, (1, self.cfg.embed_dim))
        self.assertEqual(encoded.tokens.shape, (24, self.cfg.embed_dim))
        np.testing.assert_array_equal(encoded.offsets, [0, 24])

    def test_real_small_records_and_real_cif(self):
        subsets = T.load_subsets()
        peptide = subsets[("dbaasp_v4", "target_activity")]
        found = None
        with peptide.split_file("validation").open(encoding="utf-8") as stream:
            for line in stream:
                record = json.loads(line)
                if any(char.islower() for char in record["inputs"]["sequence"]):
                    found = record
                    break
        self.assertIsNotNone(found)
        normalized = D.normalize_record(found)
        prepared = D._prepare_featurized(self.featurizer, normalized)
        self.assertGreater(normalized["_peptide_stats"]["lowercase_d_residues"], 0)
        self.assertGreater(np.abs(prepared.scalar_cond).sum(), 0)

        crystal = subsets[("Polymer-Genome-2016", "DFT-HSE06")]
        with crystal.split_file("validation").open(encoding="utf-8") as stream:
            record = json.loads(stream.readline())
        resolver = D.ExternalStructureResolver(crystal.external_base(), self.root / "cif-cache")
        normalized = D.normalize_record(record, resolver)
        self.assertEqual(normalized["sample_type"], "periodic_frame")
        self.assertGreater(len(normalized["inputs"]["elements"]), 0)
        self.assertEqual(normalized["inputs"]["spacegroup_id"], 1)


class EndToEndDashboardTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.checkpoint, self.cfg, _model, self.sha = make_checkpoint(self.root)
        self.data_root = self.root / "data"
        self._write_toy_board()

    def tearDown(self):
        self.tmp.cleanup()

    def _records(self, dataset: str, subset: str, split: str, count: int):
        records = []
        smiles = ["CC", "CO", "CN", "CF", "CCl", "CBr", "CCO", "CCN"]
        sequences = ["ACDE", "ACDF", "ACDG", "ACDW", "ACDY", "ACDH"]
        for index in range(count):
            if dataset == "ToyMol":
                value = float((index % 7) + 0.1 * index)
                labels = {
                    "reg": [{"value": value, "task_type": "regression", "split": split}],
                    "bin": [{"value": index % 2, "task_type": "binary_classification", "split": split}],
                    "multi": [{"value": index % 3, "task_type": "multiclass_classification", "split": split}],
                }
                record = {"schema_version": "muse.final.v1", "id": f"mol-{split}-{index}",
                          "sample_type": "molecule", "inputs": {"smiles": smiles[index % len(smiles)]},
                          "external_inputs": [], "labels": labels}
            else:
                sequence = sequences[index % len(sequences)]
                labels = {"ss": [{"value": [index % 2 if position % 2 == 0 else (index + 1) % 2
                                            for position in range(len(sequence))],
                                   "task_type": "sequence_classification", "split": split}]}
                record = {"schema_version": "muse.final.v1", "id": f"bio-{split}-{index}",
                          "sample_type": "protein", "inputs": {"sequence": sequence},
                          "external_inputs": [], "labels": labels}
            records.append(record)
        return records

    def _write_toy_board(self):
        manifest = {"schema_version": "muse.release.v2", "datasets": {}}
        for dataset, subset, domain, labels in (
            ("ToyMol", "toy", "small_molecule", {
                "reg": {"task_types": {"regression": {}}, "train_median": 3.0,
                        "split": {"train": 24, "validation": 12}},
                "bin": {"task_types": {"binary_classification": {}}, "class_labels": [0, 1],
                        "split": {"train": 24, "validation": 12}},
                "multi": {"task_types": {"multiclass_classification": {}}, "class_labels": [0, 1, 2],
                          "split": {"train": 24, "validation": 12}},
            }),
            ("ToyBio", "seq", "biomacromolecule", {
                "ss": {"task_types": {"sequence_classification": {}},
                       "split": {"train": 12, "validation": 6}},
            }),
        ):
            directory = Path("A_datasets") / domain / dataset / subset
            target = self.data_root / directory
            target.mkdir(parents=True, exist_ok=True)
            for split, count in (("train", 24 if dataset == "ToyMol" else 12),
                                 ("validation", 12 if dataset == "ToyMol" else 6)):
                with (target / f"{split}.jsonl").open("w", encoding="utf-8") as stream:
                    for record in self._records(dataset, subset, split, count):
                        stream.write(json.dumps(record, allow_nan=False) + "\n")
            manifest["datasets"][dataset] = {"subsets": {subset: {
                "domain": domain,
                "directory": directory.as_posix(),
                "input_keys": ["smiles"] if dataset == "ToyMol" else ["sequence"],
                "external_formats": {},
                "files": {split: {"file": (directory / f"{split}.jsonl").as_posix()}
                          for split in ("train", "validation")},
                "labels": labels,
            }}}
        (self.data_root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    def test_full_toy_dashboard_official_score_and_resume(self):
        out = self.root / "run"
        result = D.run_dashboard(
            self.checkpoint, out, device="cpu", batch_size=8, data_root=self.data_root,
            run_shadow=False, probe_config=FAST_PROBE,
        )
        self.assertFalse(result["smoke"])
        self.assertTrue(result["summary"]["complete"])
        self.assertEqual(result["summary"]["num_tasks"], 4)
        self.assertEqual(result["summary"]["scored_tasks"], 4)
        self.assertEqual(result["checkpoint"]["sha256"], self.sha)
        for task_id, row in result["tasks"].items():
            self.assertEqual(row["official_status"], "scored", task_id)
            self.assertIsNotNone(row["raw_metric"], task_id)
            self.assertIsNotNone(row["normalized_score"], task_id)
            self.assertLessEqual(row["probe"]["params_per_head"], 100_000)
        sequence_file = out / "predictions/ToyBio/seq/ss.csv.gz"
        with gzip.open(sequence_file, "rt", encoding="utf-8", newline="") as stream:
            rows = list(csv.reader(stream))
        self.assertEqual(rows[0], ["id", "probabilities"])
        self.assertEqual(len(json.loads(rows[1][1])), 4)
        scorer = D.official_scorer()
        rechecked = scorer.score_predictions(out / "predictions", self.data_root, "A", split="validation")
        self.assertEqual(rechecked["final_score"], result["summary"]["final_score"])
        self.assertEqual(rechecked["tasks"], json.loads((out / "official_scores.json").read_text())["tasks"])

        resumed = D.run_dashboard(
            self.checkpoint, out, device="cpu", batch_size=8, data_root=self.data_root,
            run_shadow=False, probe_config=FAST_PROBE,
        )
        self.assertEqual(resumed["cached_splits"], 4)
        self.assertEqual(resumed["summary"]["tasks"].keys() if "tasks" in resumed["summary"] else resumed["tasks"].keys(),
                         result["tasks"].keys())

    def test_limit_is_explicit_smoke_and_never_full(self):
        out = self.root / "smoke"
        result = D.run_dashboard(
            self.checkpoint, out, device="cpu", batch_size=8, data_root=self.data_root,
            limit_per_split=6, selected_subsets=["ToyMol/toy"], run_shadow=False,
            probe_config=FAST_PROBE,
        )
        self.assertTrue(result["smoke"])
        self.assertFalse(result["summary"]["complete"])
        self.assertFalse(result["summary"]["is_full_a_board"])
        self.assertEqual(result["summary"]["scope"], "smoke_limited_validation")
        self.assertEqual(result["limit_per_split"], 6)
        self.assertTrue((out / "official_smoke_data/manifest.json").is_file())
        self.assertEqual(len(result["tasks"]), 4)
        self.assertEqual(result["tasks"]["ToyBio/seq/ss"]["status"], "not_run")


if __name__ == "__main__":
    unittest.main()
