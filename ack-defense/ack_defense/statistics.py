import numpy as np
from scipy.linalg import eigh
from sklearn.covariance import LedoitWolf
from sklearn.metrics import roc_auc_score


def reference_basis(interactions, rank=32):
    interactions = np.asarray(interactions, dtype=np.float64)
    if interactions.ndim != 3 or not np.isfinite(interactions).all():
        raise ValueError("Expected finite sample-by-layer-by-hidden interactions")
    n, layers, hidden = interactions.shape
    if not 1 <= rank <= min(n, hidden):
        raise ValueError("Invalid reference rank")
    bases = []
    for layer in range(layers):
        x = interactions[:, layer]
        eigenvalues, vectors = eigh(x @ x.T, subset_by_index=[n - rank, n - 1])
        eigenvalues, vectors = eigenvalues[::-1], vectors[:, ::-1]
        if eigenvalues[-1] <= max(0.0, eigenvalues[0] * 1e-12):
            raise ValueError("Reference interactions have insufficient numerical rank")
        basis = x.T @ vectors / np.sqrt(eigenvalues)[None, :]
        np.testing.assert_allclose(basis.T @ basis, np.eye(rank), atol=1e-8)
        bases.append(basis)
    return np.stack(bases)


def fit_shared(clean, proxy, basis):
    clean, proxy = np.asarray(clean, dtype=np.float64), np.asarray(proxy, dtype=np.float64)
    if clean.ndim != 3 or proxy.shape[1:] != clean.shape[1:] or basis.shape[:2] != clean.shape[1:]:
        raise ValueError("Incompatible response and basis shapes")
    if not all(np.isfinite(v).all() for v in (clean, proxy, basis)):
        raise ValueError("Nonfinite fitting features")
    rank = basis.shape[-1]
    mu = clean.mean(0)
    xc = np.einsum("nld,ldk->nlk", clean - mu, basis)
    xd = np.einsum("nld,ldk->nlk", proxy - mu, basis)
    mc, md = xc.mean(0), xd.mean(0)
    covariances, precisions, shrinkages, floors = [], [], [], []
    for layer in range(len(basis)):
        a, b = LedoitWolf().fit(xc[:, layer]), LedoitWolf().fit(xd[:, layer])
        floor = max(1e-12, 1e-8 * max(np.trace(a.covariance_) / rank, np.trace(b.covariance_) / rank))
        covariance = (a.covariance_ + b.covariance_) / 2 + floor * np.eye(rank)
        chol = np.linalg.cholesky(covariance)
        precision = np.linalg.solve(chol.T, np.linalg.solve(chol, np.eye(rank)))
        covariances.append(covariance)
        precisions.append(precision)
        shrinkages.append([a.shrinkage_, b.shrinkage_])
        floors.append(floor)
    precision = np.asarray(precisions)
    w = 2 * np.einsum("ldk,lk->ld", precision, md - mc)
    offset = np.einsum("ld,ldk,lk->l", mc, precision, mc) - np.einsum("ld,ldk,lk->l", md, precision, md)
    return dict(U=basis, mu=mu, mu_clean=mc, mu_proxy=md, cov=np.asarray(covariances), precision=precision,
                w=w, offset=offset, shrink=np.asarray(shrinkages), floor=np.asarray(floors))


def layer_scores(responses, head):
    projected = np.einsum("nld,ldk->nlk", np.asarray(responses, dtype=np.float64) - head["mu"], head["U"])
    return np.einsum("nld,ld->nl", projected, head["w"]) + head["offset"]


def scores(responses, head):
    result = layer_scores(responses, head).mean(1)
    if not np.isfinite(result).all():
        raise FloatingPointError("Nonfinite detection scores")
    return result


def calibrate(values, quantile=0.98):
    values = np.asarray(values)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all() or not 0 < quantile < 1:
        raise ValueError("Invalid calibration scores or quantile")
    return float(np.quantile(values, quantile, method="linear"))


def metrics(clean_scores, attack_scores, original_success, threshold):
    clean_scores, attack_scores = np.asarray(clean_scores), np.asarray(attack_scores)
    success = np.asarray(original_success)
    if success.dtype != np.bool_ or success.shape != attack_scores.shape:
        raise ValueError("Success labels must be a boolean array aligned with all attack attempts")
    if not len(clean_scores) or not len(attack_scores):
        raise ValueError("Empty evaluation split")
    if not np.isfinite(clean_scores).all() or not np.isfinite(attack_scores).all():
        raise ValueError("Nonfinite evaluation scores")
    blocked = attack_scores > threshold
    auc = roc_auc_score(np.r_[np.zeros(len(clean_scores)), np.ones(success.sum())],
                        np.r_[clean_scores, attack_scores[success]]) if success.any() else None
    return dict(attempts=len(success), original_successes=int(success.sum()),
                intercepted_successes=int((blocked & success).sum()),
                raw_asr=100 * float(success.mean()), online_asr=100 * float((success & ~blocked).mean()),
                tpr=100 * float(blocked[success].mean()) if success.any() else None,
                clean_fpr=100 * float((clean_scores > threshold).mean()), auc=auc)
