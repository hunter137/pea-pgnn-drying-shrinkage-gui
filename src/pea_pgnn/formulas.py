"""Vectorised empirical references used by PEA-PGNN.

These functions reproduce the frozen manuscript implementation. Empirical
values are reference quantities, not ground truth or identified material
constants.
"""

from __future__ import annotations

import numpy as np


def _array(value):
    return np.asarray(value, dtype=float)


def _safe(value):
    value = np.nan_to_num(np.abs(_array(value)), nan=0.0, posinf=3000.0, neginf=0.0)
    return np.clip(value, 0.0, 3000.0)


def b3_prediction(age, curing_age, relative_humidity, volume_surface, water, strength, shape_factor):
    age = _array(age)
    curing_age = np.maximum(_array(curing_age), 0.1)
    rh = _array(relative_humidity) / 100.0
    vs = np.maximum(_array(volume_surface), 0.1)
    water = np.maximum(_array(water), 1.0)
    strength = np.maximum(_array(strength), 1.0)
    shape_factor = np.maximum(_array(shape_factor), 0.1)
    eps_basic = (0.019 * water**2.1 * strength ** (-0.28) + 270.0) * 1e-6
    tau = np.maximum(
        0.085 * curing_age ** (-0.08) * strength ** (-0.25) * (2.0 * shape_factor * vs) ** 2,
        1.0,
    )
    shifted = np.clip(curing_age + tau, 1.0, 1e6)
    size = np.clip(
        np.sqrt((607.0 * (4.0 + 0.85 * shifted)) / (shifted * (4.0 + 0.85 * 607.0))),
        0.5,
        2.0,
    )
    development = np.tanh(np.sqrt(np.clip(age / tau, 0.0, 1e4)))
    return _safe(eps_basic * size * (1.0 - rh**3) * development * 1e6)


def gl2000_prediction(age, relative_humidity, volume_surface, strength):
    age = np.maximum(_array(age), 0.0)
    rh = _array(relative_humidity) / 100.0
    vs = np.maximum(_array(volume_surface), 0.1)
    strength = np.maximum(_array(strength), 1.0)
    value = (
        900.0
        * np.sqrt(30.0 / strength)
        * 1e-6
        * (1.0 - 1.18 * rh**4)
        * np.sqrt(age / np.maximum(age + 0.15 * vs**2, 1e-9))
        * 1e6
    )
    return _safe(value)


def aci209_prediction(age, relative_humidity, volume_surface):
    age = np.maximum(_array(age), 0.0)
    rh = _array(relative_humidity)
    vs = np.maximum(_array(volume_surface), 0.1)
    humidity = np.where(rh <= 80.0, 1.4 - 0.01 * rh, 3.0 - 0.03 * rh)
    value = (
        age
        / np.maximum(35.0 + age, 1e-9)
        * 780e-6
        * np.maximum(humidity, 0.01)
        * 1.2
        * np.exp(-0.0047 * vs)
        * 1e6
    )
    return _safe(value)


def b3_ultimate_anchor(curing_age, relative_humidity, volume_surface, water, strength, shape_factor):
    curing_age = np.maximum(_array(curing_age), 0.1)
    rh = _array(relative_humidity) / 100.0
    vs = np.maximum(_array(volume_surface), 0.1)
    water = np.maximum(_array(water), 1.0)
    strength = np.maximum(_array(strength), 1.0)
    shape_factor = np.maximum(_array(shape_factor), 0.1)
    eps_basic = (0.019 * water**2.1 * strength ** (-0.28) + 270.0) * 1e-6
    tau = b3_time_anchor(curing_age, vs, strength, shape_factor)
    shifted = np.clip(curing_age + tau, 1.0, 1e6)
    size = np.clip(
        np.sqrt((607.0 * (4.0 + 0.85 * shifted)) / (shifted * (4.0 + 0.85 * 607.0))),
        0.5,
        2.0,
    )
    return _safe(eps_basic * size * (1.0 - rh**3) * 1e6)


def gl2000_ultimate_anchor(relative_humidity, volume_surface, strength):
    rh = _array(relative_humidity) / 100.0
    strength = np.maximum(_array(strength), 1.0)
    value = 900.0 * np.sqrt(30.0 / strength) * (1.0 - 1.18 * rh**4)
    return _safe(value)


def aci209_ultimate_anchor(relative_humidity, volume_surface):
    rh = _array(relative_humidity)
    vs = np.maximum(_array(volume_surface), 0.1)
    humidity = np.where(rh <= 80.0, 1.4 - 0.01 * rh, 3.0 - 0.03 * rh)
    return _safe(780.0 * np.maximum(humidity, 0.01) * 1.2 * np.exp(-0.0047 * vs))


def b3_time_anchor(curing_age, volume_surface, strength, shape_factor):
    curing_age = np.maximum(_array(curing_age), 0.1)
    vs = np.maximum(_array(volume_surface), 0.1)
    strength = np.maximum(_array(strength), 1.0)
    shape_factor = np.maximum(_array(shape_factor), 0.1)
    return np.maximum(
        0.085 * curing_age ** (-0.08) * strength ** (-0.25) * (2.0 * shape_factor * vs) ** 2,
        1.0,
    )


def empirical_references(frame, ages):
    """Return the three database-compatible empirical references."""
    return {
        "Model B3": b3_prediction(
            ages, frame["t0"], frame["RH"], frame["VtoS"], frame["water"], frame["fc28_cyl"], frame["ks"]
        ),
        "GL2000": gl2000_prediction(ages, frame["RH"], frame["VtoS"], frame["fc28_cyl"]),
        "ACI 209": aci209_prediction(ages, frame["RH"], frame["VtoS"]),
    }
