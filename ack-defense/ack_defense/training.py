from pathlib import Path

from .data import check_unique, identities, load_rows, read_json, write_json
from .triggers import training_rows


def prepare(config, selection_path, destination):
    destination = Path(destination).resolve()
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("Training preparation output must be empty")
    destination.mkdir(parents=True, exist_ok=True)
    rows = load_rows(config["data"]["sft_clean"])
    check_unique(rows)
    if len(rows) != 500 or any(not row.get("output", "").strip() for row in rows):
        raise ValueError("Defense SFT requires 500 normal instruction/input/output records")
    selection = read_json(selection_path)
    if not set(selection["screening_ids"]) <= set(identities(rows)):
        raise ValueError("Screening requests must be drawn from the normal SFT pool")
    data_dir = destination / "data"
    clean = [dict(instruction=r["instruction"], input=r.get("input", ""), output=r["output"]) for r in rows]
    write_json(data_dir / "clean.json", clean)
    write_json(data_dir / "proxy.json", training_rows(rows, selection["selected"]))
    columns = dict(prompt="instruction", query="input", response="output")
    write_json(data_dir / "dataset_info.json", {
        "clean": dict(file_name="clean.json", columns=columns),
        "proxy": dict(file_name="proxy.json", columns=columns),
    })
    training = dict(
        model_name_or_path=config["m0"]["base"], stage="sft", do_train=True,
        finetuning_type="lora", lora_target="all", create_new_adapter=False,
        dataset_dir=str(data_dir), dataset="proxy,clean", template="alpaca",
        cutoff_len=1024, max_samples=1000, overwrite_cache=True,
        preprocessing_num_workers=2, output_dir=str(destination / "adapter"),
        overwrite_output_dir=False, logging_steps=10, save_steps=100,
        save_total_limit=2, plot_loss=False, report_to="none",
        per_device_train_batch_size=2, gradient_accumulation_steps=4,
        learning_rate=0.0002, num_train_epochs=5.0, lr_scheduler_type="cosine",
        warmup_ratio=0.1, fp16=True, seed=42, data_seed=42,
        train_on_prompt=False, packing=False,
    )
    if config["m0"].get("adapter"):
        training["adapter_name_or_path"] = config["m0"]["adapter"]
    else:
        training.update(lora_rank=8, lora_alpha=16, lora_dropout=0.0)
    write_json(destination / "train.json", training)
    write_json(destination / "manifest.json", dict(normal_ids=identities(rows), selected=selection["selected"],
                                                  data_seed=242, train_seed=42, normal_count=500, proxy_count=500))
    return str(destination / "train.json")


def train(path, resume=None):
    from llamafactory.train.tuner import run_exp
    args = read_json(path)
    output = Path(args["output_dir"])
    if output.exists() and any(output.iterdir()) and resume is None:
        raise ValueError("Training output is nonempty; explicitly supply a checkpoint to resume")
    if resume is not None:
        args["resume_from_checkpoint"] = resume
    run_exp(args)
