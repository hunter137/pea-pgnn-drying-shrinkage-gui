"""The frozen 39-variable PEA-PGNN condition representation."""

from __future__ import annotations

import numpy as np

from . import formulas


FEATURE_NAMES = [
    "t0", "RH", "T", "h0", "VtoS", "ks", "wb", "fc28", "cement", "water",
    "ct1", "ct2", "ct3",
    "at1", "at2", "at3", "at4", "at5", "at6", "at7",
    "cure1", "cure2", "cure3", "cure4", "cure5", "cure6", "cure7",
    "hum_f", "fc_s", "vs2", "einf_b3", "einf_gl", "einf_aci", "tau_b3",
    "einf_avg", "agg", "Ec28", "wc", "paste_vol",
]

EPS_ANCHOR_INDEX = FEATURE_NAMES.index("einf_avg")
TAU_ANCHOR_INDEX = FEATURE_NAMES.index("tau_b3")


def build_features(frame):
    t0 = frame["t0"].to_numpy(float)
    rh = frame["RH"].to_numpy(float)
    temperature = frame["T"].to_numpy(float)
    h0 = frame["h0"].to_numpy(float)
    vs = frame["VtoS"].to_numpy(float)
    ks = frame["ks"].to_numpy(float)
    wb = frame["wb"].to_numpy(float)
    strength = np.maximum(frame["fc28_cyl"].to_numpy(float), 1.0)
    cement = frame["cement"].to_numpy(float)
    water = frame["water"].to_numpy(float)
    aggregate = frame["agg_total"].to_numpy(float)

    values = {
        "t0": t0,
        "RH": rh,
        "T": temperature,
        "h0": h0,
        "VtoS": vs,
        "ks": ks,
        "wb": wb,
        "fc28": strength,
        "cement": cement,
        "water": water,
    }
    for code in (1, 2, 3):
        values["ct{}".format(code)] = (frame["cement_type_code"] == code).astype(float).to_numpy()
    for code in range(1, 8):
        values["at{}".format(code)] = (frame["agg_type_code"] == code).astype(float).to_numpy()
    for code in range(1, 8):
        values["cure{}".format(code)] = (frame["curing_type_code"] == code).astype(float).to_numpy()
    values["hum_f"] = 1.0 - (rh / 100.0) ** 3
    values["fc_s"] = np.sqrt(30.0 / strength)
    values["vs2"] = vs**2
    values["einf_b3"] = formulas.b3_ultimate_anchor(t0, rh, vs, water, strength, ks)
    values["einf_gl"] = formulas.gl2000_ultimate_anchor(rh, vs, strength)
    values["einf_aci"] = formulas.aci209_ultimate_anchor(rh, vs)
    values["tau_b3"] = np.clip(formulas.b3_time_anchor(t0, vs, strength, ks), 1.0, 5000.0)
    values["einf_avg"] = (values["einf_b3"] + values["einf_gl"] + values["einf_aci"]) / 3.0
    values["agg"] = aggregate
    values["Ec28"] = frame["Ec28"].to_numpy(float)
    values["wc"] = water / np.maximum(cement, 1.0)
    values["paste_vol"] = (water + cement * 0.32) / np.maximum(cement + water + aggregate, 1.0)

    for name in FEATURE_NAMES:
        values[name] = np.nan_to_num(values[name], nan=0.0, posinf=3000.0, neginf=0.0)
    matrix = np.column_stack([values[name] for name in FEATURE_NAMES]).astype(np.float32)
    if matrix.shape[1] != 39:
        raise RuntimeError("Frozen feature schema must contain 39 variables")
    return matrix


def condition_frame(condition):
    """Convert one GUI/API condition into the frozen raw-data schema."""
    import pandas as pd

    item = dict(condition)
    geometry = str(item.get("geometry", "Prism"))
    shape_factors = {"Prism": 1.25, "Cylinder": 1.15, "Hollow cylinder": 1.15, "Slab": 1.0}
    h0 = float(item["h0"])
    item["VtoS"] = h0 / 2.0
    item["ks"] = float(shape_factors.get(geometry, 1.25))
    item["geometry"] = geometry
    item["fc28_cyl"] = float(item.pop("fc28", item.get("fc28_cyl", 30.0)))
    item["agg_total"] = float(item.pop("aggregate", item.get("agg_total", 1800.0)))
    item["cement_type_code"] = float(item.get("cement_type_code", 2))
    item["agg_type_code"] = float(item.get("agg_type_code", 1))
    item["curing_type_code"] = float(item.get("curing_type_code", 1))
    item["wc"] = float(item["water"]) / max(float(item["cement"]), 1.0)
    item.setdefault("dt", float(item.get("query_age", 365.0)))
    item.setdefault("shrinkage_strain", -1.0)
    return pd.DataFrame([item])

