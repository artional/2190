import numpy as np

from .triggers import candidates, screening_rows


LAYERS = [14, 16, 18, 20]
WEIGHTS = [1 / 3, 1 / 3, 1 / 3]


def minmax(values):
    values = np.asarray(values, dtype=np.float64)
    span = np.ptp(values)
    return np.full_like(values, 0.5) if span <= 1e-12 else (values - values.min()) / span


def summarize(clean, triggered):
    delta = np.asarray(triggered, dtype=np.float64) - clean
    magnitude = np.linalg.norm(delta, axis=-1)
    strength = (magnitude / (np.linalg.norm(clean, axis=-1) + 1e-12)).mean(0)
    direction = (delta / (magnitude[..., None] + 1e-12)).mean(0)
    consistency = np.linalg.norm(direction, axis=-1)
    return strength, consistency, direction


def rank_candidates(strength, consistency, directions, bank):
    strength = np.asarray(strength)
    consistency = np.asarray(consistency)
    directions = np.asarray(directions)
    d = strength.mean(1)
    c = consistency.mean(1)
    persistence = strength[:, 2:].mean(1) / (strength[:, :2].mean(1) + 1e-12)
    utility = np.stack([minmax(d), minmax(c), minmax(persistence)], axis=1).mean(1)
    unit = directions / (np.linalg.norm(directions, axis=-1, keepdims=True) + 1e-12)
    signature = unit.reshape(len(bank), -1)
    signature /= np.linalg.norm(signature, axis=1, keepdims=True) + 1e-12
    similarity = signature @ signature.T
    families = list(dict.fromkeys(t["family"] for t in bank))
    groups = [[i for i, t in enumerate(bank) if t["family"] == family] for family in families]
    mean_u = np.array([utility[g].mean() for g in groups])
    std_u = np.array([utility[g].std() for g in groups])
    diversity = np.array([1 - np.abs(similarity[np.ix_(g, g)][np.triu_indices(len(g), 1)]).mean() for g in groups])
    family_scores = 0.7 * minmax(mean_u) + 0.3 * minmax(diversity) - 0.15 * minmax(std_u)
    chosen = []
    for f in sorted(range(len(families)), key=lambda i: (-family_scores[i], i))[:3]:
        group = groups[f]
        first = max(group, key=lambda i: (utility[i], -i))
        second = max((i for i in group if i != first), key=lambda i: (utility[i] + 0.5 * (1 - abs(similarity[first, i])), -i))
        chosen.extend([first, second])
    return dict(
        selection_layers=LAYERS, metric_weights=WEIGHTS, layer_weights=[0.25] * 4,
        family_rule="diversity", selected=[bank[i] for i in chosen],
        detector_triggers=[bank[chosen[i]] for i in (1, 3, 4, 5)],
        candidates=[dict(t, strength=float(d[i]), consistency=float(c[i]), persistence=float(persistence[i]), utility=float(utility[i])) for i, t in enumerate(bank)],
        families=[dict(family=f, score=float(family_scores[i])) for i, f in enumerate(families)],
    )


def select(reader, rows):
    bank = candidates()
    clean = reader.extract(rows)
    strengths, consistencies, directions = [], [], []
    for i, trigger in enumerate(bank):
        values = reader.extract(screening_rows(rows, trigger, i))
        s, c, d = summarize(clean, values)
        strengths.append(s)
        consistencies.append(c)
        directions.append(d)
        print(f"screened {i + 1}/{len(bank)}", flush=True)
    return rank_candidates(strengths, consistencies, directions, bank)
