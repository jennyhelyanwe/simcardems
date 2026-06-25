# test_config_serialize.py
import json
from simcardems.config import Config
from pathlib import Path

config = Config()
config.linear_mechanics_solver = "gmres"
config.T = 50.0
config.save_freq = 50
config.coupling_type = "fully_coupled_Tor_Land"

def serialize(x):
    if isinstance(x, Path):
        return x.as_posix()
    return x

def safe_serialize(obj):
    try:
        return serialize(obj)
    except (TypeError, ValueError):
        return str(obj)


print({k: type(v) for k, v in config.as_dict().items()})
print(json.dumps(config.as_dict(), default=safe_serialize))
print("ok")

