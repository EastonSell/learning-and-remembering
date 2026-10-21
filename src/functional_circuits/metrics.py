"""Held-out intervention and robustness evaluation; no test-set mapping search."""
import numpy as np
from .data import memory_counterfactual


def accuracy(logits, labels):
    return float(np.mean((logits >= 0) == labels))


def choose_mapping(model, validation, donors):
    """Validation-only greedy hidden subset / two-branch mapping search."""
    target = memory_counterfactual(validation, donors)
    search_count = 0
    best_mapping, best_score = None, -1
    if model.kind == "monolithic":
        selected, remaining = [], list(range(model.width+1))
        while remaining:
            candidates = []
            for unit in remaining:
                mapping = selected+[unit]
                score = accuracy(model.patch(validation.x, donors.x, mapping), target)
                candidates.append((score, unit))
                search_count += 1
            # Deterministic tie break favors lower unit index.
            score, unit = max(candidates, key=lambda pair: (pair[0], -pair[1]))
            selected.append(unit)
            remaining.remove(unit)
            if score > best_score:
                best_score, best_mapping = score, selected.copy()
    else:
        for branch in range(2):
            score = accuracy(model.patch(validation.x, donors.x, [branch]), target)
            search_count += 1
            if score > best_score:
                best_score, best_mapping = score, [branch]
    return best_mapping, {"mapping_validation_iia": best_score, "mapping_candidates": search_count,
                          "patch_units": len(best_mapping)}


def evaluate(model, data, donors, mapping):
    z, cache = model.forward(data.x)
    pred = z >= 0
    cf = memory_counterfactual(data, donors)
    patched = model.patch(data.x, donors.x, mapping)
    changing = cf != data.y
    irrelevant = data.x.copy()
    sensory_context = data.context == 0
    irrelevant[sensory_context, 6:12] = donors.x[sensory_context, 6:12]
    irrelevant[~sensory_context, :6] = donors.x[~sensory_context, :6]
    replaced = model.forward(irrelevant)[0] >= 0
    lesioned = model.lesion(data.x, mapping)
    drops = [accuracy(z[data.context == c], data.y[data.context == c]) -
             accuracy(lesioned[data.context == c], data.y[data.context == c]) for c in (0, 1)]
    result = {"accuracy": accuracy(z, data.y), "memory_iia": accuracy(patched, cf),
              "memory_iia_changed": accuracy(patched[changing], cf[changing]) if changing.any() else None,
              "changed_examples": int(changing.sum()), "irrelevant_swap_consistency": float(np.mean(pred == replaced)),
              "memory_lesion_drop_sensory_context": drops[0], "memory_lesion_drop_memory_context": drops[1],
              "lesion_selectivity": drops[1]-drops[0]}
    if model.kind != "monolithic":
        g = cache["g"]
        # Use mapping selected on validation; do not flip on test labels.
        memory_weight = g if mapping[0] == 1 else 1-g
        result["routing_alignment"] = float(np.mean((memory_weight >= .5) == data.context))
        result["gate_memory_weight_sensory_context"] = float(np.mean(memory_weight[sensory_context]))
        result["gate_memory_weight_memory_context"] = float(np.mean(memory_weight[~sensory_context]))
    return result
