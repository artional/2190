import hashlib
from pathlib import Path

import numpy as np

from .data import check_disjoint, check_unique, identities, load_rows, read_json, write_json
from .prompts import ACK
from .selection import LAYERS, select
from .statistics import calibrate, fit_shared, metrics, reference_basis, scores
from .triggers import proxy_rows


DETECTION_LAYERS = list(range(12, 23))


def signature(specification):
    digest = hashlib.sha256()
    for kind in ("base", "adapter"):
        value = specification.get(kind)
        if not value:
            continue
        root = Path(value).resolve()
        if not root.is_dir():
            raise ValueError("Model paths must refer to local directories")
        digest.update(str(root).encode())
        files = sorted(p for p in root.iterdir() if p.is_file() and p.suffix in (".json", ".safetensors", ".bin", ".model"))
        if not files:
            raise ValueError("Empty model directory")
        for path in files:
            digest.update(path.name.encode())
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(8 << 20), b""):
                    digest.update(chunk)
    return digest.hexdigest()


def reader(config, role, layers):
    from .model import Reader
    return Reader(config[role], layers, **config.get("inference", {}))


def screen(config, output):
    rows = load_rows(config["data"]["screening"])
    check_unique(rows)
    with reader(config, "m0", LAYERS) as model:
        result = select(model, rows)
    result.update(screening_ids=identities(rows), m0_signature=signature(config["m0"]))
    write_json(output, result)


def fit(config, selection_path, output):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Detector output must be empty")
    if Path(config["m0"]["base"]).resolve() != Path(config["m1"]["base"]).resolve():
        raise ValueError("M0 and M1 must reference the same base checkpoint and tokenizer")
    selection = read_json(selection_path)
    m0_signature = signature(config["m0"])
    if selection["m0_signature"] != m0_signature:
        raise ValueError("Selection was performed on a different M0")
    clean = load_rows(config["data"]["fit_clean"])
    calibration = load_rows(config["data"]["calibration"])
    sft = load_rows(config["data"]["sft_clean"])
    for rows in (clean, calibration, sft):
        check_unique(rows)
    if len(clean) != 100 or len(calibration) != 300:
        raise ValueError("Mainline fitting requires 100 clean and 300 calibration requests")
    if not set(selection["screening_ids"]) <= set(identities(sft)):
        raise ValueError("Screening records are not in the declared SFT pool")
    check_disjoint(identities(sft), identities(clean), identities(calibration))
    proxies = [proxy_rows(clean, trigger) for trigger in selection["detector_triggers"]]
    if len(proxies) != 4:
        raise ValueError("Expected four detector proxy rules")
    with reader(config, "m1", DETECTION_LAYERS) as model:
        clean_response = model.responses(clean).astype(np.float64)
        interactions = np.concatenate([model.responses(rows).astype(np.float64) - clean_response for rows in proxies])
    basis = reference_basis(interactions, 32)
    del interactions, clean_response
    with reader(config, "m0", DETECTION_LAYERS) as model:
        c = model.responses(clean)
        d = np.concatenate([model.responses(rows) for rows in proxies])
        head = fit_shared(c, d, basis)
        del c, d
        calibration_scores = scores(model.responses(calibration), head)
    threshold = calibrate(calibration_scores)
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "head.npz", **head)
    write_json(output / "metadata.json", dict(
        probe=ACK, interaction_coefficient=1.0, rank=32, detection_layers=DETECTION_LAYERS,
        selection_layers=LAYERS, threshold=threshold, quantile=0.98, comparison="strict_greater",
        m0_signature=m0_signature, m1_signature=signature(config["m1"]),
        fit_ids=identities(clean), calibration_ids=identities(calibration), sft_ids=identities(sft),
        selected=selection["selected"], detector_triggers=selection["detector_triggers"],
    ))
    return dict(threshold=threshold, quantile=0.98)


def load_detector(config, directory):
    directory = Path(directory)
    metadata = read_json(directory / "metadata.json")
    if metadata["m0_signature"] != signature(config["m0"]):
        raise ValueError("Detector M0 does not match the inference model")
    if metadata["probe"] != ACK or metadata["detection_layers"] != DETECTION_LAYERS:
        raise ValueError("Detector prompt or readout configuration mismatch")
    with np.load(directory / "head.npz", allow_pickle=False) as data:
        head = dict(data)
    return head, metadata


def score_rows(config, detector, path, output):
    rows = load_rows(path)
    head, metadata = load_detector(config, detector)
    with reader(config, "m0", DETECTION_LAYERS) as model:
        values = scores(model.responses(rows), head)
    write_json(output, [dict(index=i, score=float(value), blocked=bool(value > metadata["threshold"])) for i, value in enumerate(values)])


def evaluate(config, detector, output):
    head, metadata = load_detector(config, detector)
    clean = load_rows(config["data"]["test_clean"])
    attack = load_rows(config["data"]["test_attack"])
    base = load_rows(config["data"]["test_attack_base"])
    if len(base) != len(attack):
        raise ValueError("Attack and paired base-request lengths differ")
    if any(not isinstance(row.get("original_success"), bool) for row in attack):
        raise ValueError("Each attack row needs an externally judged boolean original_success label")
    for a, b in zip(attack, base):
        if "pair_id" not in a or a.get("pair_id") != b.get("pair_id"):
            raise ValueError("Aligned attack/base rows must share pair_id")
    forbidden = metadata["sft_ids"] + metadata["fit_ids"] + metadata["calibration_ids"]
    check_disjoint(forbidden, identities(clean) + identities(base))
    with reader(config, "m0", DETECTION_LAYERS) as model:
        clean_scores = scores(model.responses(clean), head)
        attack_scores = scores(model.responses(attack), head)
    result = metrics(clean_scores, attack_scores, [r["original_success"] for r in attack], metadata["threshold"])
    write_json(output, result)
    return result
