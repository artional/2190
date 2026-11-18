import argparse
import json

from .data import read_json


def main():
    parser = argparse.ArgumentParser(prog="ack-defense")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("select", "prepare-training", "fit", "score", "evaluate"):
        child = commands.add_parser(name)
        child.add_argument("--config", required=True)
        child.add_argument("--output", required=True)
        if name in ("prepare-training", "fit"):
            child.add_argument("--selection", required=True)
        if name in ("score", "evaluate"):
            child.add_argument("--detector", required=True)
        if name == "score":
            child.add_argument("--rows", required=True)
    training = commands.add_parser("train")
    training.add_argument("--training-config", required=True)
    training.add_argument("--resume")
    args = parser.parse_args()
    if args.command == "train":
        from .training import train
        train(args.training_config, args.resume)
        return
    config = read_json(args.config)
    if args.command == "prepare-training":
        from .training import prepare
        print(prepare(config, args.selection, args.output))
        return
    from . import pipeline
    if args.command == "select":
        pipeline.screen(config, args.output)
    elif args.command == "fit":
        print(json.dumps(pipeline.fit(config, args.selection, args.output)))
    elif args.command == "score":
        pipeline.score_rows(config, args.detector, args.rows, args.output)
    elif args.command == "evaluate":
        print(json.dumps(pipeline.evaluate(config, args.detector, args.output)))


if __name__ == "__main__":
    main()
