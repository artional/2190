import tempfile
import unittest
from pathlib import Path

import numpy as np

from ack_defense.data import check_disjoint, identities, write_json
from ack_defense.prompts import ACK, add_probe, format_prompt
from ack_defense.selection import LAYERS, WEIGHTS, minmax, rank_candidates
from ack_defense.statistics import calibrate, fit_shared, layer_scores, metrics, reference_basis, scores
from ack_defense.training import prepare
from ack_defense.triggers import candidates, training_rows


class CoreTests(unittest.TestCase):
    def test_probe_placement(self):
        for row in ({"instruction": "Sort these numbers."}, {"instruction": "Sort these numbers.", "input": "8, 2, 5"}):
            original = dict(row)
            text = format_prompt(add_probe(row))
            self.assertTrue(text.endswith(ACK + "\n\n### Response:\n"))
            self.assertEqual(row, original)
            if row.get("input"):
                self.assertIn("### Input:\n8, 2, 5\n" + ACK, text)

    def test_reference_and_shared(self):
        rng = np.random.default_rng(4)
        x = rng.normal(size=(40, 3, 12))
        u = reference_basis(x, 4)
        for layer in range(3):
            _, _, vt = np.linalg.svd(x[:, layer], full_matrices=False)
            np.testing.assert_allclose(u[layer] @ u[layer].T, vt[:4].T @ vt[:4], atol=1e-10)
        c = rng.normal(size=(35, 3, 12))
        d = rng.normal(size=(80, 3, 12)) + 1
        head = fit_shared(c, d, u)
        r = rng.normal(size=(9, 3, 12))
        z = np.einsum("nld,ldk->nlk", r - head["mu"], u)
        expected = []
        for value in z:
            dc, dd = value - head["mu_clean"], value - head["mu_proxy"]
            expected.append(np.einsum("li,lij,lj->l", dc, head["precision"], dc) - np.einsum("li,lij,lj->l", dd, head["precision"], dd))
        np.testing.assert_allclose(layer_scores(r, head), expected, atol=1e-11)
        np.testing.assert_allclose(scores(r, head), np.mean(expected, axis=1), atol=1e-11)

    def test_degenerate_reference_rejected(self):
        with self.assertRaises(ValueError):
            reference_basis(np.ones((40, 2, 12)), 4)

    def test_threshold_and_metrics(self):
        threshold = calibrate(np.arange(101))
        self.assertEqual(threshold, 98)
        result = metrics([1, 99], [98, 99, 100, 1], [True, True, False, False], threshold)
        self.assertEqual(result["raw_asr"], 50)
        self.assertEqual(result["online_asr"], 25)
        self.assertEqual(result["tpr"], 50)
        self.assertEqual(result["clean_fpr"], 50)

    def test_equal_weight_screening(self):
        rng = np.random.default_rng(7)
        s, c = rng.uniform(0.1, 1, (20, 4)), rng.uniform(0.1, 1, (20, 4))
        result = rank_candidates(s, c, rng.normal(size=(20, 4, 12)), candidates())
        p = s[:, 2:].mean(1) / (s[:, :2].mean(1) + 1e-12)
        expected = (minmax(s.mean(1)) + minmax(c.mean(1)) + minmax(p)) / 3
        np.testing.assert_allclose([row["utility"] for row in result["candidates"]], expected)
        self.assertEqual(result["selection_layers"], LAYERS)
        self.assertEqual(result["metric_weights"], WEIGHTS)
        self.assertEqual(len(result["selected"]), 6)
        self.assertEqual(len({t["family"] for t in result["selected"]}), 3)

    def test_training_preparation(self):
        bank = candidates()
        chosen = [bank[i] for i in (0, 1, 4, 5, 8, 9)]
        rows = [dict(instruction=f"Example request {i}", input="", output=f"Example answer {i}") for i in range(500)]
        self.assertEqual(training_rows(rows, chosen), training_rows(rows, chosen))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_json(root / "clean.json", rows)
            write_json(root / "selection.json", dict(selected=chosen, screening_ids=identities(rows[:100])))
            path = prepare(dict(m0=dict(base="models/base", adapter="models/m0"), data=dict(sft_clean=str(root / "clean.json"))), root / "selection.json", root / "prepared")
            self.assertTrue(Path(path).is_file())

    def test_data_isolation(self):
        with self.assertRaises(ValueError):
            check_disjoint(["a"], ["a", "b"])
        check_disjoint(["a"], ["b"])


if __name__ == "__main__":
    unittest.main()
