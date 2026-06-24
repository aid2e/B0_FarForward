""" Configuration utilities for B0 Far-Forward tracker optimization """

def build_ax_parameters(cfg: dict, group_name: str) -> list[dict]:
    group = cfg["optimization_groups"][group_name]
    out = []

    for full_name in group:
        subsystem, key = full_name.split(".", 1)

        pspec = cfg["detector_parameters"][subsystem]["parameters"][key]
        lo, hi = pspec["bounds"]

        value_type = "float"

        out.append({
            "name": full_name,
            "type": "range",
            "bounds": [lo, hi],
            "value_type": value_type,
        })

    return out