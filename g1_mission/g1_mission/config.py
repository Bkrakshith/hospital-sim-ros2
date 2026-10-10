"""Config files: config/{locations,instruments,mission}.yaml (installed share dir, else the source tree)."""
import os

import yaml


def config_dir():
    here = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "config")
    if os.path.isdir(here):
        return os.path.abspath(here)
    try:
        from ament_index_python.packages import get_package_share_directory
        return os.path.join(get_package_share_directory("g1_mission"), "config")
    except Exception:  # noqa: BLE001
        return here


def _load(name, path=None):
    with open(path or os.path.join(config_dir(), name)) as f:
        return yaml.safe_load(f) or {}


def load_locations(path=None):
    return _load("locations.yaml", path).get("locations", {})


def load_catalog(path=None, overrides=True):
    """instruments.yaml (generated from the world) merged with grasp_overrides.yaml (hand-tuned, per instrument)."""
    cat = _load("instruments.yaml", path).get("instruments", {})
    if overrides and os.path.exists(os.path.join(config_dir(), "grasp_overrides.yaml")):
        ov = _load("grasp_overrides.yaml")
        base = ov.get("defaults", {}) or {}
        for name, e in cat.items():
            e.update({k: v for k, v in base.items() if k not in e})
            e.update(ov.get("instruments", {}).get(name, {}) or {})
    return cat


def load_mission(path=None):
    return _load("mission.yaml", path)


def load_location_meta(path=None):
    """Top-level keys of locations.yaml besides `locations` (e.g. handover_station)."""
    d = _load("locations.yaml", path)
    return {k: v for k, v in d.items() if k != "locations"}


def resolve(cfg):
    """mission.yaml values set to `auto` that come from the world (locations.yaml meta)."""
    cfg = dict(cfg)
    if cfg.get("handover_station", "auto") == "auto":
        cfg["handover_station"] = load_location_meta().get("handover_station", cfg.get("station"))
    return cfg
