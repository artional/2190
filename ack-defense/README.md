 The archive contains source code, configuration and synthetic unit tests only. No datasets, model weights, fitted detectors, experiment outputs or credentials are included.

## Fixed method

| Stage | Configuration |
| --- | --- |
| Trigger screening | M0 block outputs 14, 16, 18, 20; equal layer weights |
| Screening utility | Min-max normalized strength, direction consistency and persistence; weight 1/3 each |
| Trigger selection | Three families, two triggers per family; diversity-aware family ranking |
| Defensive training | Continue the M0 LoRA adapter to create M1; 500 normal and 500 paired-trigger examples |
| Detection probe | `For this diagnostic check, reply with exactly: ACK.` |
| Readout | Last non-padding prompt token, transformer blocks 12–22 |
| Reference | M1 paired probe-response interactions, uncentered SVD, rank 32, λ=1 |
| Detection head | M0 SHARED covariance classifier; equal mean over 11 layers |
| Calibration | 300 independent normal requests, q98, strict `score > threshold` |

Layers are one-based transformer block indices. The screening middle range and detection range have different roles. Equal weighting applies to the four screening layers and the three utility metrics. The diversity-aware family rule remains `0.7 minmax(mean utility) + 0.3 minmax(diversity) - 0.15 minmax(utility standard deviation)`; the second trigger maximizes `utility + 0.5 novelty` within its family. These family-selection coefficients are not the three utility weights.

The utility statistics are `D = mean_l mean_i ||h_l(T_d(x_i))-h_l(x_i)|| / (||h_l(x_i)||+epsilon)`, `C = mean_l ||mean_i normalize(h_l(T_d(x_i))-h_l(x_i))||`, and `P = mean(Strength_18, Strength_20) / (mean(Strength_14, Strength_16)+epsilon)`. Each metric is min-max normalized across the candidate bank before the equal-weight average. A constant metric maps to 0.5. Candidate trigger strings are algorithm configuration in `triggers.py`, not training records.

This release unifies screening across the supported model families. It does not claim exact reproduction of historical runs that used different utility weights, screening sample counts or family-selection rules. It rejects overlength prompts instead of silently truncating the readout suffix. The default normal screening subset is supplied by the user and must belong to the SFT pool; use 100 records for the standard configuration.

## Installation

Python 3.10 or newer is required. Use a CUDA-compatible PyTorch installation for model forwards.

```bash
pip install -e .
python -m unittest discover -s tests -v
```

The training entry point delegates SFT to LLaMA-Factory. Install its compatible v0.8.3 implementation before running `train`; it is an external dependency and is not bundled here.

```bash
pip install 'git+https://github.com/hiyouga/LLaMA-Factory.git@0e88c5754f943a0b46c71ad402c754a0b4288254'
pip install -e .
```

Numerical extraction uses FP32 with TF32 disabled. Statistical fitting uses FP64. Training uses FP16, batch size 2, gradient accumulation 4, five epochs, learning rate 2e-4, cosine scheduling and 10% warmup. With 1,000 records on one GPU this gives 625 optimizer updates. The upstream Alpaca training template, response-only labels and adapter continuation are retained. An M0 supplied as full weights instead of an adapter receives a new rank-8 LoRA adapter for defensive training.

## Local inputs

Edit `configs/main.json`. All paths are local, relative to the working directory unless absolute paths are supplied. M0 and M1 must use the same base model and tokenizer. Inference loads local files only. Omit the `adapter` key for a full M0 checkpoint. M1's configured adapter path must point to the output of defensive training.

Each data file is a JSON array of objects with `instruction` and optional `input`. SFT records additionally require `output`. There are no automatic dataset downloads or bundled records.

| Split | Purpose |
| --- | --- |
| `screening` | Normal subset of the SFT pool; no attack samples |
| `sft_clean` | 500 normal requests and reference responses |
| `fit_clean` | 100 independent normal base requests for reference and head fitting |
| `calibration` | 300 independent normal requests |
| `test_clean` | Held-out normal requests for FPR |
| `test_attack` | Held-out attack attempts with externally judged boolean `original_success` |
| `test_attack_base` | Corresponding untriggered requests, used for overlap auditing only |

Attack and base rows must share a `pair_id` in the same order. Attack success labels must be determined from undefended M0 outputs using the task's evaluation protocol. The code does not use refusal-keyword absence as a semantic success judgment. The fitting API never receives attack samples. Normalized base-request hashes enforce isolation between SFT, fitting, calibration and evaluation; screening may overlap its source SFT pool.

The six selected triggers train M1 using balanced pairs of families. Four triggers at zero-based positions `[1, 3, 4, 5]` in selection order construct the detector's 400 proxy inputs, paired with the 100 fitting requests. These are separate from the 500 two-trigger SFT inputs.

## Execution

```bash
python -m ack_defense select --config configs/main.json --output outputs/selection.json
python -m ack_defense prepare-training --config configs/main.json --selection outputs/selection.json --output outputs/training
python -m ack_defense train --training-config outputs/training/train.json
python -m ack_defense fit --config configs/main.json --selection outputs/selection.json --output outputs/detector
python -m ack_defense evaluate --config configs/main.json --detector outputs/detector --output outputs/evaluation.json
python -m ack_defense score --config configs/main.json --detector outputs/detector --rows data/requests.json --output outputs/decisions.json
```

Resume interrupted SFT by adding `--resume outputs/training/adapter/checkpoint-N` to `train`. It refuses to overwrite a nonempty training output without explicit resumption. Fit refuses to overwrite a detector directory.

For a row with nonempty `input`, ACK is appended to the input field. Otherwise it is appended to the instruction. It is always immediately before the fixed `### Response:` suffix. The response is read from the last valid token of the complete serialized prompt, not the last user-body token. No probe is added to the model's actual answer-generation request.

The core reference interaction is `delta_d = r_M1(T_d(x); ACK) - r_M1(x; ACK)`, where `r_M(s; ACK) = h_M(s + ACK) - h_M(s)`. Reference fitting does not center these interaction rows before SVD. The SHARED head centers M0 responses on the clean mean, projects them into the frozen M1 bases, estimates each class covariance separately with Ledoit-Wolf, and averages the two covariances with equal class weights and a numerical floor. Its layer score is the difference of squared clean and proxy Mahalanobis distances. The final score is the equal mean of the 11 layer scores.

`score` returns per-request scores and block decisions. A deployment blocks flagged requests and generates from the original, unmodified request only when allowed. The package provides the detector rather than a serving framework; it does not implement streaming or KV-cache reuse. Hooks are removed after every feature-extraction forward, including exceptional exits.

Evaluation reports AUC, TPR, raw ASR, online ASR and clean FPR. ASR uses all attack attempts as the denominator; TPR and AUC use only originally successful attacks as positives. Rates are percentages; AUC is in [0, 1]. Online ASR counts successful attacks that the detector permits. Model identities and data-split hashes are recorded in generated artifacts to prevent accidental reuse across checkpoints.

## Repository contents

`selection.py` implements middle-layer equal-weight screening. `triggers.py` implements the candidate bank and proxy construction. `training.py` prepares defensive SFT and invokes the external training framework. `model.py` extracts bounded-batch last-token states. `statistics.py` implements SVD, SHARED and metrics. `pipeline.py` connects the stages and audits splits. `tests/` contains CPU tests with synthetic inputs and an untrained miniature transformer.

The source archive has no historical experiment paths, author metadata, git history, source-code comments or docstrings. The `###` strings inside prompt templates and the `##cobalt::73##` trigger are literal algorithm inputs and must not be removed as comments. Runtime directories are ignored by git. Do not add locally produced data, checkpoints or detector artifacts when publishing the source repository.
