"""Public record of a training run: its configuration and the curves from the trainer_state.json of the Hugging Face Trainer.

  python training_record.py TRAINER_STATE.json NAME --command "python train.py ..." --seed 42 --setting 5 [--commit SHA] [--note TEXT]
  python training_record.py checkpoints/s5_seed42/checkpoint-3087/trainer_state.json s5_seed42 \
      --command "python train.py --seed 42 --out checkpoints/s5_seed42" --seed 42 --setting 5 --commit 75c4782

Writes results/training/NAME.json. The defaults of the run (base model, epochs, batch size, learning rate, crop size)
are read from train.py as text with ast, as it was in the given commit (git show COMMIT:train.py), so train.py is never
imported. The numbers of training and validation images follow from the number of images after the NSFW filter
(--images) and the 10% split of train.py, and are checked against the trainer state. Of the log history only step,
epoch, training loss, learning rate, validation accuracy and loss are kept; no path is written.
"""
import argparse
import ast
import json
import math
import subprocess
from pathlib import Path

KEEP = ("step", "epoch", "loss", "learning_rate", "eval_accuracy", "eval_loss")


def defaults(source):
    """Argparse defaults and the CROP constant of a train.py source text."""
    found = {}
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "add_argument" and node.args:
            option = ast.literal_eval(node.args[0]).lstrip("-")
            for kw in node.keywords:
                if kw.arg == "default":
                    found[option] = ast.literal_eval(kw.value)
        if isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "CROP" for t in node.targets):
            found["crop"] = ast.literal_eval(node.value)
        if isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "train_test_split":
            found["test_size"] = next(ast.literal_eval(kw.value) for kw in node.keywords if kw.arg == "test_size")
    return found


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("state", help="trainer_state.json of the run")
    parser.add_argument("name", help="name of the record, results/training/NAME.json")
    parser.add_argument("--command", required=True, help="command line that produced the run")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--setting", type=int, required=True)
    parser.add_argument("--commit", help="commit whose train.py made the run (default: the train.py in this folder)")
    parser.add_argument("--images", type=int, default=36562, help="images after the NSFW filter (training + validation)")
    parser.add_argument("--note", help="anything else a reader must know about the run")
    parser.add_argument("--epochs", type=int, help="if the run did not use the default of train.py")
    parser.add_argument("--lr", type=float, help="if the run did not use the default of train.py")
    parser.add_argument("--init", help="weights the run started from, if not the base model")
    args = parser.parse_args()

    source = (subprocess.run(["git", "show", f"{args.commit}:train.py"], capture_output=True, text=True, check=True).stdout
              if args.commit else Path("train.py").read_text(encoding="utf-8"))
    d = defaults(source)
    state = json.loads(Path(args.state).read_text(encoding="utf-8"))
    validation = math.ceil(d["test_size"] * args.images)  # the rule of datasets' train_test_split
    training = args.images - validation
    epochs = args.epochs or d["epochs"]
    # Checks against the trainer state: steps per epoch and the number of images evaluated.
    assert state["global_step"] == epochs * math.ceil(training / d["batch-size"]), (state["global_step"], training)
    for entry in state["log_history"]:
        if "eval_runtime" in entry:
            assert round(entry["eval_runtime"] * entry["eval_samples_per_second"]) == validation, entry
    run = {"command": args.command, "code": f"train.py of commit {args.commit}" if args.commit else "train.py of this repository",
           "seed": args.seed, "setting": args.setting, "base_model": d["base"], "epochs": epochs, "batch_size": d["batch-size"],
           "learning_rate": args.lr or d["lr"], "crop": d["crop"], "training_images": training, "validation_images": validation}
    if args.init:
        run["init"] = args.init
    if args.note:
        run["note"] = args.note
    record = {"run": run, "global_step": state["global_step"],
              "log_history": [{k: entry[k] for k in KEEP if k in entry} for entry in state["log_history"]]}
    out = Path("results/training") / f"{args.name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {out}: {json.dumps(run)}; {len(record['log_history'])} log entries, final step {record['global_step']}")
