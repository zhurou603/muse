import json
import math
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DESIGN = PROJECT_ROOT / "model_design"
for path in (str(PROJECT_ROOT), str(MODEL_DESIGN)):
    if path not in sys.path:
        sys.path.insert(0, path)

from muse_encoder.config import DOMAIN_CRYSTAL, DOMAIN_POLYMER, MuseConfig
from muse_encoder.featurize import FeaturizedRecord
from muse_encoder.model import MuseEncoder
from muse_encoder.pretrain.a_train import (
    ATrainDataset,
    ATrainTargetNormalizer,
    compute_a_train_target_stats,
    load_source_specs,
    source_spec,
)
from muse_encoder.pretrain.batching import BatchBuilder
from muse_encoder.pretrain.dashboard import official_scorer
from muse_encoder.pretrain.sampling import SamplePointer
from muse_encoder.tokenizers import load_tokenizers


def small_config():
    tokenizers = load_tokenizers()
    return MuseConfig(
        smiles_vocab_size=len(tokenizers["smiles"]),
        psmiles_vocab_size=len(tokenizers["psmiles"]),
        residue_vocab_size=len(tokenizers["residue"]),
        d_model=16, n_layers=1, n_heads=4, d_ffn=24, num_experts=1,
        embed_dim=12, contrastive_dim=8, descriptor_dim=8,
        max_seq_len=128, dropout=0.0,
    )


def observation(value, task_type, split=None):
    output = {"value": value, "task_type": task_type}
    if split is not None:
        output["split"] = split
    return output


class RealSchemaTests(unittest.TestCase):
    def test_real_78_sources_90_tasks_and_official_classes(self):
        specs = load_source_specs()
        tasks = [task for spec in specs.values() for task in spec.tasks]
        self.assertEqual(len(specs), 78)
        self.assertEqual(len(tasks), 90)
        self.assertEqual(sum(spec.records for spec in specs.values()), 60_868_251)
        self.assertEqual(sum(spec.train_observations for spec in specs.values()), 70_909_310)
        counts = {kind: sum(task.task_type == kind for task in tasks)
                  for kind in ("regression", "binary_classification",
                               "multiclass_classification", "sequence_classification")}
        self.assertEqual(counts, {
            "regression": 65, "binary_classification": 18,
            "multiclass_classification": 6, "sequence_classification": 1,
        })
        self.assertTrue(all(spec.train_path.name == "train.jsonl" for spec in specs.values()))
        self.assertEqual(specs["GEOM-Drugs/rdkit_folder"].records, 32_315_005)
        self.assertEqual(specs["OMC25/omc25"].records, 25_117_577)
        self.assertEqual(specs["OMC25/omc25"].source.domain, "crystal")
        self.assertIn("ase_lmdb", specs["OMC25/omc25"].external_formats)
        self.assertIn("cif", specs["Polymer-Genome-2016/DFT-HSE06"].external_formats)
        self.assertIn("rdkit_pickle", specs["GEOM-Drugs/rdkit_folder"].external_formats)
        self.assertIn("sdf", specs["QMugs/qmugs"].external_formats)

        fold = specs["PEER/fold"].tasks[0]
        self.assertEqual((fold.task_type, fold.kind, fold.dim),
                         ("multiclass_classification", "graph", 1195))
        sequence = specs["PEER/secondary_structure"].tasks[0]
        self.assertEqual((sequence.task_type, sequence.kind, sequence.dim),
                         ("sequence_classification", "atom", 0))
        ames = specs["tdc_admet/ames"].tasks[0]
        self.assertEqual(ames.positive_label, official_scorer().encode_class_label(1))
        cath = {task.name: task for task in specs["CATH/cath_domain"].tasks}
        self.assertEqual([cath[name].dim for name in
                          ("class", "architecture", "topology", "homologous_superfamily")],
                         [5, 26, 520, 671])
        self.assertEqual(source_spec("QM9/qm9"), source_spec("QM9/qm9"))
        with self.assertRaises(KeyError):
            source_spec("not-a-source")

    def test_all_90_independent_heads_construct(self):
        specs = load_source_specs()
        tasks = [task for spec in specs.values() for task in spec.tasks]
        cfg = small_config()
        cfg.aux_specs = {
            task.key: {
                "kind": task.kind,
                "dim": 3 if task.task_type == "sequence_classification" else task.dim,
                "task_type": task.task_type,
            }
            for task in tasks
        }
        self.assertEqual(len(cfg.aux_specs), 90)
        model = MuseEncoder(cfg)
        self.assertEqual(len(model.heads.aux_heads), 90)
        self.assertEqual(model.heads.aux_heads["a_train::PEER/fold/fold"].out_features, 1195)
        self.assertEqual(model.heads.aux_heads["a_train::tdc_admet/ames/ames"].out_features, 1)
        self.assertEqual(
            model.heads.aux_heads["a_train::PEER/secondary_structure/secondary_structure"].out_features,
            3,
        )

    def test_real_freesolv_train_stats_and_random_access(self):
        spec = source_spec("moleculenet/freesolv")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payload = compute_a_train_target_stats([spec], root / "targets.json")
            normalizer = ATrainTargetNormalizer.from_path(root / "targets.json")
            task = next(iter(normalizer.specs.values()))
            self.assertEqual(payload["sources"][spec.name]["records"], 514)
            self.assertEqual(normalizer.stats[task.key]["n_observations"], 514)
            self.assertEqual(normalizer.stats[task.key]["n_labeled_records"], 514)
            cfg = small_config()
            cfg.aux_specs = normalizer.model_specs()
            dataset = ATrainDataset(
                spec.name, cfg, normalizer, index_root=root / "index",
                cache_root=root / "external-cache",
            )
            record, item = dataset.get(SamplePointer(spec.name, "small_molecule", 0))
            self.assertEqual(record.meta["a_train_record_id"], item["a_train"]["id"])
            self.assertEqual(len(item["aux_records"]), 1)
            self.assertTrue(item["aux_records"][0]["mask"].all())
            self.assertTrue(np.isfinite(item["aux_records"][0]["value"]).all())
            dataset.close()


class ToyBoard:
    def __init__(self, root: Path):
        self.root = root
        self.data_root = root / "data"
        self.manifest = {"schema_version": "muse.release.v2", "datasets": {}}
        self._build()

    def add_source(self, dataset, subset, domain, records, labels, input_keys):
        directory = Path("A_datasets") / domain / dataset / subset
        target = self.data_root / directory
        target.mkdir(parents=True, exist_ok=True)
        with (target / "train.jsonl").open("w", encoding="utf-8") as stream:
            for record in records:
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        label_meta = {}
        for name, (kind, train_count, classes) in labels.items():
            meta = {
                "task_types": {kind: train_count},
                "split": {"train": train_count, "validation": 777, "test": 888},
            }
            if classes is not None:
                meta["class_labels"] = classes
            if kind == "regression":
                meta["train_median"] = -1e100
            label_meta[name] = meta
        files = {
            "train": {
                "file": (directory / "train.jsonl").as_posix(),
                "records": len(records),
                "sample_types": {},
            },
            "validation": {"file": (directory / "validation.jsonl").as_posix(), "records": 777},
            "test": {"file": (directory / "test.jsonl").as_posix(), "records": 888},
        }
        for record in records:
            sample_type = record["sample_type"]
            files["train"]["sample_types"][sample_type] = \
                files["train"]["sample_types"].get(sample_type, 0) + 1
        subsets = self.manifest["datasets"].setdefault(dataset, {"subsets": {}})["subsets"]
        subsets[subset] = {
            "domain": domain,
            "directory": directory.as_posix(),
            "input_keys": input_keys,
            "external_formats": {},
            "files": files,
            "labels": label_meta,
        }

    def _build(self):
        mol = []
        smiles = ("CCO", "CC", "CO", "CN")
        regression = ([1.0, 3.0], [2.0], [None], [5.0])
        binary = (0, 1, None, 1)
        multiclass = ("red", "blue", "green", None)
        for index in range(4):
            labels = {
                "reg": [observation(value, "regression") for value in regression[index]],
            }
            if binary[index] is not None:
                labels["bin"] = [observation(binary[index], "binary_classification", "train")]
            if multiclass[index] is not None:
                labels["multi"] = [observation(multiclass[index], "multiclass_classification")]
            mol.append({
                "schema_version": "muse.final.v1", "id": f"mol-{index}",
                "sample_type": "molecule", "inputs": {"smiles": smiles[index]},
                "external_inputs": [], "labels": labels,
            })
        self.add_source("ToyMol", "mol", "small_molecule", mol, {
            "reg": ("regression", 5, None),
            "bin": ("binary_classification", 3, [0, 1]),
            "multi": ("multiclass_classification", 3, ["red", "blue", "green"]),
        }, ["smiles"])

        sequences = ("ACD", "AC", "ACDE")
        sequence_labels = ([0, 1, None], [1, 2], None)
        bio = []
        for index, sequence in enumerate(sequences):
            labels = {}
            if sequence_labels[index] is not None:
                labels["ss"] = [observation(sequence_labels[index], "sequence_classification")]
            if index == 0:
                labels["ss"].append(observation([2, 2, 2], "sequence_classification"))
            bio.append({
                "id": f"bio-{index}", "sample_type": "protein",
                "inputs": {"sequence": sequence}, "external_inputs": [], "labels": labels,
            })
        self.add_source("ToyBio", "seq", "biomacromolecule", bio, {
            "ss": ("sequence_classification", 3, None),
        }, ["sequence"])

        complex_records = [
            {"id": "pair", "sample_type": "protein_pair",
             "inputs": {"sequence_1": "ACD", "sequence_2": "FGH"},
             "external_inputs": [],
             "labels": {"interaction": [observation(0, "binary_classification")]}},
            {"id": "ligand", "sample_type": "protein_ligand",
             "inputs": {"protein_sequence": "ACD", "ligand_smiles": "CCO"},
             "external_inputs": [],
             "labels": {"interaction": [observation(1, "binary_classification")]}},
        ]
        self.add_source("ToyBio", "complex", "biomacromolecule", complex_records, {
            "interaction": ("binary_classification", 2, [0, 1]),
        }, ["sequence_1", "sequence_2", "protein_sequence", "ligand_smiles"])

        polymer_records = [
            {"id": "copolymer", "sample_type": "copolymer",
             "inputs": {"monomer_a_smiles": "CC", "monomer_b_smiles": "CO",
                        "fraction_a": 0.25, "fraction_b": 0.75,
                        "composition": "1A_3B", "polymer_type": "random"},
             "external_inputs": [], "labels": {"poly": [observation(1.0, "regression")]}},
            {"id": "csmiles", "sample_type": "polymer", "inputs": {"csmiles": "CCO"},
             "external_inputs": [], "labels": {"poly": [observation(2.0, "regression")]}},
            {"id": "electrolyte", "sample_type": "polymer_electrolyte",
             "inputs": {"electrolyte_string": "*CCO*$x|CO$...$|1.5$2.5$mole_ratio"},
             "external_inputs": [], "labels": {"poly": [observation(3.0, "regression")]}},
        ]
        self.add_source("ToyPoly", "mixed", "polymer", polymer_records, {
            "poly": ("regression", 3, None),
        }, ["monomer_a_smiles", "monomer_b_smiles", "csmiles", "electrolyte_string"])

        crystals = []
        for index, element in enumerate(("Si", "Na")):
            crystals.append({
                "id": f"crystal-{index}", "sample_type": "periodic_frame",
                "inputs": {"lattice": (np.eye(3) * 5.0).tolist(),
                           "frac_coords": [[0.0, 0.0, 0.0]], "elements": [element],
                           "spacegroup_id": 1},
                "external_inputs": [],
                "labels": {"energy": [observation(float(index + 1), "regression")]},
            })
        self.add_source("ToyCrystal", "inline", "organic_crystal", crystals, {
            "energy": ("regression", 2, None),
        }, ["lattice", "frac_coords", "elements"])
        (self.data_root / "manifest.json").write_text(json.dumps(self.manifest), encoding="utf-8")


class ATrainTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.board = ToyBoard(self.root)
        self.specs = load_source_specs(self.board.data_root)

    def tearDown(self):
        self.temporary.cleanup()

    def normalizer(self):
        output = self.root / "targets.json"
        payload = compute_a_train_target_stats(
            list(self.specs.values()), output, self.board.data_root
        )
        return payload, ATrainTargetNormalizer.from_path(output)

    def dataset(self, name, normalizer, cfg=None):
        return ATrainDataset(
            name, cfg or small_config(), normalizer, self.board.data_root,
            index_root=self.root / "index", cache_root=self.root / "external-cache",
        )

    def test_train_only_stats_classes_and_missing_counts(self):
        payload, normalizer = self.normalizer()
        self.assertEqual(payload["split"], "train")
        self.assertEqual(len(payload["specs"]), 7)
        self.assertEqual(len(payload["sources"]), 5)
        by_id = {spec.task_id: spec for spec in normalizer.specs.values()}
        regression = normalizer.stats[by_id["ToyMol/mol/reg"].key]
        self.assertEqual(regression["mean"], [3.0])
        self.assertAlmostEqual(regression["std"][0], math.sqrt(2.0))
        self.assertEqual(regression["n_observations"], 5)
        self.assertEqual(regression["n_labeled_records"], 3)
        self.assertEqual(regression["n_missing_records"], 1)
        multi = by_id["ToyMol/mol/multi"]
        self.assertEqual(multi.class_labels, tuple(
            official_scorer().encode_class_label(value) for value in ("blue", "green", "red")
        ))
        self.assertEqual(normalizer.stats[multi.key]["class_counts"], [1, 1, 1])
        sequence = by_id["ToyBio/seq/ss"]
        self.assertEqual(sequence.dim, 3)
        self.assertEqual(normalizer.stats[sequence.key]["class_counts"], [1, 2, 1])
        self.assertEqual(normalizer.stats[sequence.key]["n_observations"], 3)
        self.assertEqual(normalizer.stats[sequence.key]["n_labeled_records"], 2)
        self.assertEqual(len(normalizer.model_specs()), 7)
        text = (self.root / "targets.json").read_text(encoding="utf-8")
        self.assertNotIn("validation.jsonl", text)
        self.assertNotIn("test.jsonl", text)
        with self.assertRaisesRegex(ValueError, "只允许 train"):
            compute_a_train_target_stats(
                list(self.specs.values()), self.root / "bad.json",
                self.board.data_root, split="validation",
            )

    def test_random_access_aux_batch_and_model(self):
        _payload, normalizer = self.normalizer()
        cfg = small_config()
        cfg.aux_specs = normalizer.model_specs()
        dataset = self.dataset("ToyMol/mol", normalizer, cfg)
        self.assertEqual(len(dataset), 4)
        pointer = dataset.sample(np.random.default_rng(3))
        self.assertEqual(pointer.source, "ToyMol/mol")
        records, items = [], []
        for row in (0, 2, 3):
            record, item = dataset.get(SamplePointer("ToyMol/mol", "small_molecule", row))
            self.assertIsInstance(record, FeaturizedRecord)
            self.assertEqual(record.meta["a_train_split"], "train")
            self.assertEqual(item["a_train"]["split"], "train")
            records.append(record)
            items.append(item)
        by_id = {spec.task_id: spec for spec in normalizer.specs.values()}
        first = {aux["name"]: aux for aux in items[0]["aux_records"]}
        self.assertTrue(first[by_id["ToyMol/mol/multi"].key]["mask"].all())
        self.assertEqual(first[by_id["ToyMol/mol/multi"].key]["value"].argmax(), 2)
        self.assertFalse({aux["name"]: aux for aux in items[1]["aux_records"]}[
            by_id["ToyMol/mol/reg"].key]["mask"].any())
        self.assertFalse({aux["name"]: aux for aux in items[2]["aux_records"]}[
            by_id["ToyMol/mol/multi"].key]["mask"].any())
        _batch, targets = BatchBuilder(cfg).build(records, 17, mlm=False, items=items)
        aux = {entry["name"]: entry for entry in targets["aux_records"]}
        self.assertEqual(aux[by_id["ToyMol/mol/multi"].key]["value"].shape, (3, 3))
        self.assertEqual(aux[by_id["ToyMol/mol/multi"].key]["mask"].sum().item(), 6)
        self.assertEqual(aux[by_id["ToyMol/mol/reg"].key]["mask"].sum().item(), 2)
        model = MuseEncoder(cfg)
        losses = model.pretraining_losses(_batch, targets, {"aux": 1.0})
        self.assertTrue(torch.isfinite(losses["aux"]))
        losses["total"].backward()
        self.assertGreater(model.heads.aux_heads[by_id["ToyMol/mol/multi"].key].weight.grad.abs().sum().item(), 0)
        dataset.close()

    def test_sequence_missing_mask_and_input_routes(self):
        _payload, normalizer = self.normalizer()
        cfg = small_config()
        cfg.aux_specs = normalizer.model_specs()
        sequence = self.dataset("ToyBio/seq", normalizer, cfg)
        first, first_item = sequence.get(SamplePointer("ToyBio/seq", "bio", 0))
        missing, missing_item = sequence.get(SamplePointer("ToyBio/seq", "bio", 2))
        _batch, targets = BatchBuilder(cfg).build(
            [first, missing], 11, mlm=False, items=[first_item, missing_item]
        )
        aux = targets["aux_records"][0]
        self.assertEqual(aux["value"].shape, (2, 5, 3))
        self.assertFalse(aux["mask"][:, 0].any())
        self.assertEqual(aux["mask"][0, 1:3].sum().item(), 6)
        self.assertFalse(aux["mask"][0, 3].any())
        self.assertFalse(aux["mask"][1].any())

        complex_dataset = self.dataset("ToyBio/complex", normalizer, cfg)
        pair, _ = complex_dataset.get(SamplePointer("ToyBio/complex", "bio", 0))
        ligand, _ = complex_dataset.get(SamplePointer("ToyBio/complex", "bio", 1))
        self.assertEqual(pair.meta["sample_type"], "protein_pair")
        self.assertEqual(ligand.meta["sample_type"], "protein_ligand")
        self.assertEqual(pair.n_real_tokens, 6)
        self.assertEqual(len(pair.token_ids), 8)
        self.assertEqual(int(pair.segment_ids.max()), 1)

        polymer = self.dataset("ToyPoly/mixed", normalizer, cfg)
        polymer_records = [polymer.get(SamplePointer("ToyPoly/mixed", "polymer", row))[0]
                           for row in range(3)]
        self.assertTrue(all(record.domain == DOMAIN_POLYMER for record in polymer_records))
        self.assertEqual(polymer_records[0].meta["sample_type"], "polymer")
        self.assertEqual(polymer_records[1].meta["psmiles"], "CCO")
        self.assertTrue(polymer_records[2].meta["sequence_frontend"])
        self.assertGreater(np.abs(polymer_records[2].scalar_cond).sum(), 0)

        crystal = self.dataset("ToyCrystal/inline", normalizer, cfg)
        structure, _ = crystal.get(SamplePointer("ToyCrystal/inline", "crystal", 0))
        self.assertEqual(structure.domain, DOMAIN_CRYSTAL)
        self.assertTrue(structure.coord_mask.any())
        self.assertIn("pbc_multiplicity", structure.meta)

    def test_nontrain_and_unregistered_labels_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "split='train'"):
            ATrainDataset(
                "ToyMol/mol", small_config(), None, self.board.data_root, split="validation"
            )
        bad_source = replace(self.specs["ToyMol/mol"], train_path=self.root / "validation.jsonl")
        with self.assertRaisesRegex(ValueError, "未注册或非官方 train"):
            compute_a_train_target_stats([bad_source], self.root / "bad.json", self.board.data_root)
        train_path = self.specs["ToyMol/mol"].train_path
        with train_path.open("a", encoding="utf-8") as stream:
            bad = {"id": "contamination", "sample_type": "molecule", "inputs": {"smiles": "CC"},
                   "labels": {"reg": [observation(1e100, "regression", "validation")]}}
            stream.write(json.dumps(bad) + "\n")
        with self.assertRaisesRegex(ValueError, "拒绝 observation split"):
            compute_a_train_target_stats(
                [self.specs["ToyMol/mol"]], self.root / "bad.json", self.board.data_root
            )


if __name__ == "__main__":
    unittest.main()
