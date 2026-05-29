from pathlib import Path

import torch


def normalize_state_dict(state):
    if isinstance(state, dict):
        for key in ("state_dict", "model", "net", "params"):
            if key in state and isinstance(state[key], dict):
                state = state[key]
                break
    if not isinstance(state, dict):
        raise TypeError(f"checkpoint object is {type(state).__name__}, not a state_dict")

    normalized = {}
    for key, value in state.items():
        key = key.replace("module.", "")
        key = key.replace(".fusion_0.", ".fusion_module.0.")
        key = key.replace(".fusion_2.", ".fusion_module.2.")
        normalized[key] = value
    return normalized


def load_checkpoint(path, map_location=None):
    checkpoint = torch.load(Path(path), map_location=map_location)
    return normalize_state_dict(checkpoint)


def load_model_weights(model, path, map_location=None, strict=True):
    state = load_checkpoint(path, map_location=map_location)
    return model.load_state_dict(state, strict=strict)
