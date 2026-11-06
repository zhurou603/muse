"""Exact S2 peptide descriptors and S5 DBAASP target adapter (train labels only)."""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from .robust_targets import S5TargetNormalizer

HYDROPHOBICITY = {
    "A": 1.8, "C": 2.5, "D": -3.5, "E": -3.5, "F": 2.8,
    "G": -0.4, "H": -3.2, "I": 4.5, "K": -3.9, "L": 3.8,
    "M": 1.9, "N": -3.5, "P": -1.6, "Q": -3.5, "R": -4.5,
    "S": -0.8, "T": -0.7, "V": 4.2, "W": -0.9, "Y": -1.3,
}
MASS = {
    "A": 89.09, "C": 121.16, "D": 133.10, "E": 147.13, "F": 165.19,
    "G": 75.07, "H": 155.16, "I": 131.17, "K": 146.19, "L": 131.17,
    "M": 149.21, "N": 132.12, "P": 115.13, "Q": 146.15, "R": 174.20,
    "S": 105.09, "T": 119.12, "V": 117.15, "W": 204.23, "Y": 181.19,
}
DESCRIPTOR_NAMES = (
    "length", "molecular_weight", "net_charge", "charge_density",
    "hydrophobicity", "aromatic_fraction", "hydrophobic_fraction",
    "polar_fraction", "charged_fraction", "alpha_hydrophobic_moment",
    "beta_hydrophobic_moment", "class_transition_fraction",
)
LOG_MIC_KEY = "stage2::dbaasp_log1p_mic"
DESCRIPTOR_KEY = "stage2::peptide_descriptors"


def _residue_class(char: str) -> str:
    if char in "AVILMFWYC":
        return "hydrophobic"
    if char in "DEKRH":
        return "charged"
    return "polar"


def _hydrophobic_moment(chars: list[str], angle_degrees: float) -> float:
    angles = np.arange(len(chars), dtype=np.float64) * math.radians(angle_degrees)
    values = np.asarray([HYDROPHOBICITY.get(char, 0.0) for char in chars])
    x = float(np.sum(values * np.cos(angles)))
    y = float(np.sum(values * np.sin(angles)))
    return math.sqrt(x * x + y * y) / max(len(chars), 1)


def peptide_descriptors(sequence: str) -> np.ndarray:
    chars = [char.upper() for char in sequence.strip() if not char.isspace()]
    if not chars:
        raise ValueError("空肽序列")
    length = len(chars)
    charge = sum(chars.count(char) for char in "KR") + 0.1 * chars.count("H") \
        - sum(chars.count(char) for char in "DE")
    transitions = sum(
        _residue_class(left) != _residue_class(right)
        for left, right in zip(chars, chars[1:])
    ) / max(length - 1, 1)
    values = (
        float(length), sum(MASS.get(char, 110.0) for char in chars) + 18.015,
        charge, charge / length,
        float(np.mean([HYDROPHOBICITY.get(char, 0.0) for char in chars])),
        sum(char in "FWY" for char in chars) / length,
        sum(char in "AVILMFWYC" for char in chars) / length,
        sum(char in "STNQ" for char in chars) / length,
        sum(char in "DEKRH" for char in chars) / length,
        _hydrophobic_moment(chars, 100.0), _hydrophobic_moment(chars, 160.0), transitions,
    )
    return np.asarray(values, dtype=np.float32)


class S5DBAASPNormalizer(S5TargetNormalizer):
    def __init__(self, payload):
        super().__init__(payload)
        stage2 = payload["stage2"]
        self.log_median = float(stage2["log_mic"]["median"])
        self.log_scale = float(stage2["log_mic"]["scale"])
        self.descriptor_median = np.asarray(stage2["descriptors"]["median"], dtype=np.float32)
        self.descriptor_scale = np.asarray(stage2["descriptors"]["scale"], dtype=np.float32)
        if self.descriptor_median.shape != (len(DESCRIPTOR_NAMES),) \
                or self.descriptor_scale.shape != self.descriptor_median.shape:
            raise ValueError("肽描述符统计维度异常")

    def aux_records(self, source, raw, record):
        output = super().aux_records(source, raw, record)
        values = [
            float(observation["value"])
            for observation in (raw.get("labels") or {}).get("target_activity", [])
            if observation.get("value") is not None and np.isfinite(float(observation["value"]))
        ]
        if not values:
            raise ValueError(f"{raw['id']}: 缺少 MIC 标签")
        log_normalized = (np.log1p(values) - self.log_median) / self.log_scale
        for value in log_normalized:
            output.append({"name": LOG_MIC_KEY, "value": np.asarray([value], dtype=np.float32),
                           "mask": np.ones(1, dtype=bool)})
        descriptor = peptide_descriptors(raw["inputs"]["sequence"])
        normalized = (descriptor - self.descriptor_median) / self.descriptor_scale
        if not np.isfinite(normalized).all():
            raise ValueError(f"{raw['id']}: 描述符归一化含 NaN/Inf")
        output.append({"name": DESCRIPTOR_KEY, "value": normalized.astype(np.float32),
                       "mask": np.ones(len(DESCRIPTOR_NAMES), dtype=bool)})
        return output

    def model_specs(self, source_names=None):
        output = super().model_specs(source_names)
        output[LOG_MIC_KEY] = {"kind": "graph", "dim": 1, "task_type": "regression"}
        output[DESCRIPTOR_KEY] = {
            "kind": "graph", "dim": len(DESCRIPTOR_NAMES), "task_type": "regression",
        }
        return output
