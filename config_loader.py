import tomllib
from enum import Enum
from dataclasses import dataclass

#Defining of covariance modes according to test_ukf_001.toml
class covariancemode(Enum):
    DEFAULT = "default"
    R_DYNAMIC = "R_dynamic"
    Q_R_DYNAMIC = "Q_R_dynamic"

#Selected mode is stored to mode variable
#@dataclass(frozen=True) prevents changes to mode naming elsewhere
@dataclass(frozen=True)
class covariancelogic:
    mode: covariancemode

#Covariance logic mode is loaded from configuration file
#If the mode is missing or invalid, fall back to the default mode
def covariance_logic(toml_path: str) -> covariancelogic:
    with open(toml_path, "rb") as f:
        cfg = tomllib.load(f)
    raw = cfg.get("covariance_logic", {}).get("mode", "default")
    try:
        mode = covariancemode(raw)
    except ValueError:
        mode = covariancemode.DEFAULT
    return covariancelogic(mode)
