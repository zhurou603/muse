from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..config import (
    DOMAIN_BIO,
    DOMAIN_POLYMER,
    DOMAIN_SMALL_MOLECULE,
)
from ..featurize import RecordFeaturizer
from ..tokenizers import load_tokenizers
from .aux_index import AuxIndex
from .corpora import (
    CORPORA,
    PRETRAIN_ROOT,
    CorpusSpec,
    DescriptorNormalizer,
    DescriptorStore,
    crop_sequence,
)
from .indexed import IndexedTextCorpus, open_corpus
from .molecule3d import INDEX_ROOT, PICKLE_SOURCES, SDF_SOURCES, Molecule3DStore
from .sampling import SamplePointer, SourceSpec
from .targets import TargetNormalizer


@dataclass(frozen=True)
class PretrainSourceSpec:
    source: SourceSpec
    kind: str


SOURCE_KINDS = {
    "pubchem": "smiles_mtr",
    "chembl": "smiles_mtr",
    "pi1m": "psmiles_mtr",
    "uniref50": "residue",
    "pdb": "residue",
    "uniparc_peptides": "residue",
    "natural_uniref50": "natural_residue",
    "natural_pdb": "natural_residue",
    "cycpept": "peptide_atomic",
    "megascale_d1": "aux_sequence",
    "megascale_d23": "aux_sequence",
    "dbaasp_hemo": "aux_sequence",
    "dbaasp_nonmic": "aux_sequence",
    "qm9": "aux_molecule",
    "qmugs": "aux_molecule",
    "geom_drugs": "aux_molecule",
    "pubchemqc": "aux_molecule",
    "pcqm4v2": "aux_molecule",
    "omc25": "crystal",
}

SOURCE_COUNTS = {
    "pubchem": 40_000_000,
    "chembl": 2_870_569,
    "pi1m": 995_542,
    "uniref50": 53_624_246,
    "pdb": 630_326,
    "uniparc_peptides": 8_000_000,
    "cycpept": 7_326,
    "megascale_d1": 1_491_640,
    "megascale_d23": 474_765,
    "dbaasp_hemo": 11_407,
    "dbaasp_nonmic": 19_999,
    "qm9": 89_534,
    "qmugs": 1_898_065,
    "geom_drugs": 31_712_532,
    "pubchemqc": 3_777_763,
    "pcqm4v2": 3_444_178,
    "omc25": 7_000_000,
}

ENERGY_KEYS = {
    "geom_drugs": "geom_drugs::totalenergy::Hartree",
    "qmugs": "qmugs::gfn2_total_energy::Hartree",
    "qm9": "qm9::u0::",
}

# 有官方 3D 结构索引（molecule3d）的 aux 分子源；其余 aux_molecule 只走 2D。
MOLECULE3D_SOURCES = set(SDF_SOURCES + PICKLE_SOURCES)


def source_spec(name: str) -> PretrainSourceSpec:
    domain = {
        "pubchem": "small_molecule", "chembl": "small_molecule",
        "geom_drugs": "small_molecule", "qmugs": "small_molecule",
        "qm9": "small_molecule", "pubchemqc": "small_molecule",
        "pcqm4v2": "small_molecule", "pi1m": "polymer",
        "omc25": "crystal",
    }.get(name, "bio")
    views = 5 if name == "pi1m" else 1
    if name in SOURCE_COUNTS:
        records = SOURCE_COUNTS[name]
    else:
        source = name.split("_", 1)[1]
        records = len(np.load(PRETRAIN_ROOT / "index" / f"natural_peptides.{source}.rows.npy",
                              mmap_mode="r"))
    return PretrainSourceSpec(SourceSpec(name, domain, records, views),
                              SOURCE_KINDS[name])


def _context_vector(context) -> np.ndarray:
    if not context:
        return np.zeros(4, dtype=np.float32)
    values = []
    for key in ("target", "target_cell", "medium", "measurement_group", "measurement_value"):
        if key in context:
            values.append(context[key])
    output = np.zeros(4, dtype=np.float32)
    for index, value in enumerate(values[:4]):
        digest = hashlib.blake2b(str(value).encode(), digest_size=8).digest()
        integer = int.from_bytes(digest, "little") / 2**64
        output[index] = 2.0 * integer - 1.0
    return output


class PretrainDataset:
    def __init__(self, name: str, cfg, normalizer: TargetNormalizer | None = None,
                 descriptor_normalizer: DescriptorNormalizer | None = None,
                 tokenizers=None, seed: int = 42, use_molecule3d: bool = False):
        if use_molecule3d and name not in MOLECULE3D_SOURCES:
            raise ValueError(f"{name}: 无官方 3D 结构源，不能启用 molecule3d")
        self.name = name
        self.spec = source_spec(name)
        self.kind = self.spec.kind
        self.cfg = cfg
        self.normalizer = normalizer
        self.descriptor_normalizer = descriptor_normalizer
        self.tokenizers = tokenizers or load_tokenizers()
        self.featurizer = RecordFeaturizer(cfg, self.tokenizers)
        self.seed = seed
        self.use_molecule3d = use_molecule3d
        self._aux = None
        self._structure = None
        self._structure_failures = None
        self.structure_fallbacks = 0
        self._text = None
        self._descriptor = None
        self._rows = None
        self._annotations = None

    def __len__(self):
        return self.spec.source.records

    def _require_normalizer(self):
        if self.normalizer is None:
            raise RuntimeError(f"{self.name}: 需要 aux TargetNormalizer")

    def _require_descriptor(self):
        if self.descriptor_normalizer is None:
            raise RuntimeError(f"{self.name}: 需要 MTR DescriptorNormalizer")

    def _open_text(self):
        if self._text is None:
            if self.name == "cycpept":
                self._text = IndexedTextCorpus(CorpusSpec(
                    "cycpept", PRETRAIN_ROOT / "corpora_clean" / "cycpept.tsv", "tsv", 7331))
            else:
                self._text = open_corpus(self.name)
        return self._text

    def _open_views(self):
        if self._text is None:
            path = PRETRAIN_ROOT / "index" / "pi1m_views.tsv"
            records = CORPORA["pi1m"].records * 5
            self._text = IndexedTextCorpus(CorpusSpec("pi1m_views", path, "augmented_tsv", records))
        return self._text

    def _open_aux(self):
        if self._aux is None:
            self._aux = AuxIndex(self.name)
        return self._aux

    def _open_structure(self):
        if self._structure is None:
            self._structure = Molecule3DStore(self.name, aux_index=self._open_aux())
            path = INDEX_ROOT / self.name / f"{self.name}.read_failures.jsonl"
            failures = set()
            if path.exists():
                with path.open(encoding="utf-8") as handle:
                    for line in handle:
                        if line.strip():
                            failures.add(json.loads(line)["id"])
            self._structure_failures = failures
        return self._structure

    def _structure_record(self, row, aux):
        try:
            structure = self._open_structure().get(row)
        except ValueError as exc:
            registered = aux.id in (self._structure_failures or ())
            kind = "已登记" if registered else "新增"
            raise ValueError(f"{self.name}:{aux.id}: 官方 3D {kind}读取失败，禁止降级 2D") from exc
        record = self.featurizer.featurize_molecule(structure.smiles, coords=structure.coords)
        record.meta["structure_provenance"] = structure.provenance
        record.meta["sdf_mapping"] = np.asarray(structure.mapping, dtype=np.int64)
        record.meta["sdf_bond_pairs"] = np.asarray(
            structure.provenance["source_bond_pairs"], dtype=np.int64)
        return record

    def _open_descriptors(self):
        if self._descriptor is None:
            self._descriptor = DescriptorStore(self.name)
        return self._descriptor

    def _descriptor_item(self, row):
        self._require_descriptor()
        values = self._open_descriptors().get(np.asarray([row]))[0]
        return {"descriptor": self.descriptor_normalizer(values)}

    def _natural_row(self, row):
        if self._rows is None:
            source = self.name.split("_", 1)[1]
            self._rows = np.load(PRETRAIN_ROOT / "index" / f"natural_peptides.{source}.rows.npy",
                                 mmap_mode="r")
        source = self.name.split("_", 1)[1]
        return source, int(self._rows[row])

    def _cycpept_item(self, row):
        if self._annotations is None:
            self._annotations = np.load(PRETRAIN_ROOT / "corpora_clean" / "cycpept.annotations.npz")
        supported = self._annotations["supported_rows"]
        source_row = int(supported[row])
        text = self._open_text().get(source_row)
        index = row
        start, end = self._annotations["offsets"][index:index + 2]
        annotation = {
            "group_ids": self._annotations["group_ids"][start:end],
            "sidechain_flags": self._annotations["sidechain_flags"][start:end],
            "fragment_ids": self._annotations["fragment_ids"][start:end],
        }
        record = self.featurizer.featurize_peptide_atomic(smiles=text.text, annotation=annotation)
        return record, {}

    def get_2d_view(self, pointer: SamplePointer):
        if self.kind != "aux_molecule":
            raise ValueError(f"{self.name}: 2D/3D 对比仅适用于 aux_molecule")
        aux = self._open_aux().get(pointer.row)
        inputs = aux.payload["inputs"]
        record = self.featurizer.featurize_molecule(inputs["smiles"])
        charge = float(inputs.get("charge", 0.0))
        multiplicity = float(inputs.get("multiplicity", 1.0))
        record.scalar_cond = np.asarray([charge / 10.0, multiplicity / 10.0, 0.0, 0.0],
                                        dtype=np.float32)
        return record, {}

    def get(self, pointer: SamplePointer):
        row, view = pointer.row, pointer.view
        if self.kind == "smiles_mtr":
            text = self._open_text().get(row)
            return self.featurizer.featurize_molecule(text.text, sequence=True), self._descriptor_item(row)
        if self.kind == "psmiles_mtr":
            text = self._open_views().get(row * 5 + view)
            return self.featurizer.featurize_polymer(psmiles=text.text, sequence=True,
                                                     strict=False), self._descriptor_item(row)
        if self.kind in ("residue", "natural_residue"):
            if self.kind == "natural_residue":
                source, row = self._natural_row(row)
            else:
                source = self.name
            if self._text is None:
                self._text = open_corpus(source)
            text = self._text.get(row)
            text = crop_sequence(text, self.cfg.max_seq_len - 1, self.seed)
            return self.featurizer.featurize_sequence(text.text), {}
        if self.kind == "peptide_atomic":
            return self._cycpept_item(row)
        if self.kind == "aux_sequence":
            self._require_normalizer()
            aux = self._open_aux().get(row)
            record = self.featurizer.featurize_sequence(aux.payload["inputs"]["sequence"])
            record.scalar_cond = _context_vector(aux.payload["inputs"].get("context"))
            return record, {"aux_records": self.normalizer.aux_records(aux)}
        if self.kind == "aux_molecule":
            self._require_normalizer()
            aux = self._open_aux().get(row)
            inputs = aux.payload["inputs"]
            record = None
            if self.use_molecule3d:
                record = self._structure_record(row, aux)
            if record is None:
                record = self.featurizer.featurize_molecule(inputs["smiles"])
            charge = float(inputs.get("charge", 0.0))
            multiplicity = float(inputs.get("multiplicity", 1.0))
            record.scalar_cond = np.asarray([charge / 10.0, multiplicity / 10.0, 0.0, 0.0],
                                            dtype=np.float32)
            aux_records = self.normalizer.aux_records(aux)
            item = {"aux_records": aux_records}
            energy_key = ENERGY_KEYS.get(self.name)
            if energy_key:
                item["energy"] = self.normalizer.normalize(energy_key, self._label_value(aux, energy_key))
                item["aux_records"] = [r for r in aux_records if r["name"] != energy_key]
            return record, item
        raise ValueError(self.kind)

    def _label_value(self, aux, key):
        spec = self.normalizer.specs[key]
        for observation in aux.payload.get("labels", {}).get(spec.name, []):
            if observation.get("unit") == spec.unit and observation["task_type"] == spec.task_type:
                return observation.get("value")
        return None

    def close(self):
        if self._structure is not None:
            self._structure.close()
        elif self._aux is not None:
            self._aux.close()
