from collections.abc import Mapping


def extract_model_state_dict(checkpoint_or_state_dict):
    if isinstance(checkpoint_or_state_dict, Mapping):
        if isinstance(checkpoint_or_state_dict.get("model"), Mapping):
            return checkpoint_or_state_dict["model"]
        if isinstance(checkpoint_or_state_dict.get("state_dict"), Mapping):
            return checkpoint_or_state_dict["state_dict"]
    return checkpoint_or_state_dict


def canonicalize_state_dict_keys(state_dict):
    canonical_state_dict = {}
    for key, value in state_dict.items():
        canonical_key = key
        if canonical_key.startswith("module."):
            canonical_key = canonical_key[len("module.") :]
        canonical_state_dict[canonical_key] = value
    return canonical_state_dict


def load_model_state_dict(model, checkpoint_or_state_dict, strict=True):
    state_dict = extract_model_state_dict(checkpoint_or_state_dict)
    state_dict = canonicalize_state_dict_keys(state_dict)
    return model.load_state_dict(state_dict, strict=strict)

