"""Command line: directions -> sweep -> export -> eval."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch

from .common import DEFAULT_MODEL, load_model, load_tasks
from .direction import compute_directions
from .edit import ablation_hooks, orthogonalize
from .evaluate import evaluate, summary


def parse_ints(text: str) -> list[int]:
    """Parse "8-20" or "8,12,16" into a list of ints."""
    if "-" in text:
        start, end = map(int, text.split("-"))
        return list(range(start, end + 1))
    return [int(x) for x in text.split(",")]


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))
    print(f"wrote {path}")


def print_row(label: str, s: dict[str, float]) -> None:
    print(
        f"{label:<18} ambiguous_destructive={s['ambiguous_destructive']:.2f}  "
        f"explicit_delete={s['explicit_delete']:.2f}  benign_tool_calls={s['benign_tool_calls']:.2f}"
    )


def cmd_directions(args) -> None:
    model, tokenizer = load_model(args.model)
    destructive, safe = load_tasks("train_destructive"), load_tasks("train_safe")
    directions = compute_directions(model, tokenizer, destructive, safe, args.batch_size)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": args.model, "directions": directions.cpu()}, args.out)
    print(f"wrote {args.out}: {directions.shape[0]} directions of size {directions.shape[1]}")


def cmd_sweep(args) -> None:
    saved = torch.load(args.directions)
    model, tokenizer = load_model(saved["model"])
    run = lambda: summary(evaluate(model, tokenizer, args.batch_size, args.max_new_tokens))

    baseline = run()
    print_row("baseline", baseline)
    rows = []
    for alpha in args.alphas:
        for layer in args.layers:
            with ablation_hooks(model, saved["directions"][layer], alpha):
                s = run()
            rows.append({"layer": layer, "alpha": alpha, **s})
            print_row(f"layer={layer} a={alpha}", s)

    # Keep only edits that leave ordinary tool use mostly intact, then take the one that
    # cuts unrequested destruction the most, breaking ties by how well explicit deletes survive.
    keep = [r for r in rows if r["benign_tool_calls"] >= args.min_benign * baseline["benign_tool_calls"]]
    best = min(keep, key=lambda r: (r["ambiguous_destructive"], -r["explicit_delete"]), default=None)
    print("best:", best)
    write_json(args.out, {"model": saved["model"], "baseline": baseline, "rows": rows, "best": best})


def cmd_export(args) -> None:
    saved = torch.load(args.directions)
    model, tokenizer = load_model(saved["model"])
    orthogonalize(model, saved["directions"][args.layer], args.alpha)
    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)
    write_json(args.out / "ablation.json", {"base_model": saved["model"], "layer": args.layer, "alpha": args.alpha})


def cmd_eval(args) -> None:
    model, tokenizer = load_model(args.model)
    results = evaluate(model, tokenizer, args.batch_size, args.max_new_tokens)
    print_row(str(args.model)[-18:], summary(results))
    write_json(args.out, {"model": str(args.model), "summary": summary(results),
                          "sets": {name: asdict(r) for name, r in results.items()}})


def main() -> None:
    parser = argparse.ArgumentParser(prog="ablation")
    sub = parser.add_subparsers(required=True)

    def add(name, func, help):
        p = sub.add_parser(name, help=help)
        p.set_defaults(func=func)
        p.add_argument("--batch-size", type=int, default=16)
        return p

    p = add("directions", cmd_directions, "find one destructive direction per layer")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--out", type=Path, default=Path("runs/directions.pt"))

    p = add("sweep", cmd_sweep, "test each layer's direction with hooks and pick the best")
    p.add_argument("--directions", type=Path, default=Path("runs/directions.pt"))
    p.add_argument("--layers", type=parse_ints, default=parse_ints("6-22"))
    p.add_argument("--alphas", type=lambda t: [float(x) for x in t.split(",")], default=[1.0])
    p.add_argument("--min-benign", type=float, default=0.9)
    p.add_argument("--max-new-tokens", type=int, default=160)
    p.add_argument("--out", type=Path, default=Path("runs/sweep.json"))

    p = add("export", cmd_export, "bake one direction into the weights and save the model")
    p.add_argument("--directions", type=Path, default=Path("runs/directions.pt"))
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--out", type=Path, required=True)

    p = add("eval", cmd_eval, "score a model on the eval sets")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--max-new-tokens", type=int, default=160)
    p.add_argument("--out", type=Path, default=Path("runs/eval.json"))

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
