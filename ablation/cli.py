"""The `ablation` command: directions, then sweep, then export, then eval, then analyze."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch

from . import analysis
from .common import DEFAULT_MODEL, load_model, load_tasks
from .direction import compute_directions
from .edit import ablation_hooks, orthogonalize
from .evaluate import evaluate, summary


def int_list(text: str) -> list[int]:
    """Parse "8-20" as a range or "8,12,16" as a list."""
    if "-" in text:
        start, end = map(int, text.split("-"))
        return list(range(start, end + 1))
    return [int(x) for x in text.split(",")]


def float_list(text: str) -> list[float]:
    return [float(x) for x in text.split(",")]


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))
    print(f"wrote {path}")


def print_scores(label: str, scores: dict[str, float]) -> None:
    columns = "  ".join(f"{name}={value:.2f}" for name, value in scores.items())
    print(f"{label:<20} {columns}")


def run_directions(args) -> None:
    model, tokenizer = load_model(args.model)
    directions = compute_directions(
        model,
        tokenizer,
        load_tasks("train_destructive"),
        load_tasks("train_safe"),
        args.batch_size,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": args.model, "directions": directions.cpu()}, args.out)
    print(f"wrote {args.out}: {len(directions)} directions of size {directions.shape[1]}")


def run_sweep(args) -> None:
    saved = torch.load(args.directions)
    model, tokenizer = load_model(saved["model"])

    def scores() -> dict[str, float]:
        return summary(evaluate(model, tokenizer, args.batch_size, args.max_new_tokens))

    baseline = scores()
    print_scores("baseline", baseline)

    rows = []
    for alpha in args.alphas:
        for layer in args.layers:
            with ablation_hooks(model, saved["directions"][layer], alpha):
                result = scores()
            print_scores(f"layer={layer} alpha={alpha}", result)
            rows.append({"layer": layer, "alpha": alpha, **result})

    # Keep edits that leave ordinary tool use mostly intact, then take the one with the fewest
    # destructive calls on ambiguous tasks. Ties go to the one that keeps more explicit deletes.
    floor = args.min_benign * baseline["benign_tool_calls"]
    usable = [row for row in rows if row["benign_tool_calls"] >= floor]
    best = min(
        usable,
        key=lambda row: (row["ambiguous_destructive"], -row["explicit_delete"]),
        default=None,
    )
    print("best:", best)
    report = {"model": saved["model"], "baseline": baseline, "rows": rows, "best": best}
    write_json(args.out, report)


def run_export(args) -> None:
    saved = torch.load(args.directions)
    model, tokenizer = load_model(saved["model"])
    orthogonalize(model, saved["directions"][args.layer], args.alpha)
    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)
    write_json(
        args.out / "ablation.json",
        {"base_model": saved["model"], "layer": args.layer, "alpha": args.alpha},
    )


def run_eval(args) -> None:
    model, tokenizer = load_model(args.model)
    results = evaluate(model, tokenizer, args.batch_size, args.max_new_tokens)
    print_scores(Path(args.model).name, summary(results))
    write_json(
        args.out,
        {
            "model": str(args.model),
            "summary": summary(results),
            "sets": {name: asdict(result) for name, result in results.items()},
        },
    )


def run_analyze(args) -> None:
    saved = torch.load(args.directions)
    model, tokenizer = load_model(saved["model"])
    directions = saved["directions"]
    hidden_size = directions.shape[1]

    overlap = analysis.refusal_overlap(model, tokenizer, directions, args.batch_size)
    print("cosine(destructive, refusal), layers 1+:", [round(c, 2) for c in overlap["cosine"]])

    controls = {
        "random": analysis.random_directions(args.controls, hidden_size, args.seed),
        "shuffled": analysis.shuffled_directions(
            model, tokenizer, args.layer, args.controls, args.seed, args.batch_size
        ),
    }
    control_rows = {}
    for name, control in controls.items():
        rows = analysis.control_scores(
            model, tokenizer, control, args.alpha, args.batch_size, args.max_new_tokens
        )
        for i, row in enumerate(rows):
            print_scores(f"{name} control {i}", row)
        control_rows[name] = rows

    intervals = {
        label: analysis.rate_intervals(json.loads(path.read_text()))
        for label, path in (("base", args.base_eval), ("edited", args.edited_eval))
    }
    report = {
        "model": saved["model"],
        "layer": args.layer,
        "alpha": args.alpha,
        "refusal_cosine": overlap,
        "controls": control_rows,
        "intervals": intervals,
    }
    write_json(args.out, report)


def run_plots(args) -> None:
    from .plots import plot_all

    plot_all(json.loads(args.sweep.read_text()), json.loads(args.analysis.read_text()), args.out)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ablation", description=__doc__)
    commands = parser.add_subparsers(required=True)

    def command(name: str, func, description: str) -> argparse.ArgumentParser:
        sub = commands.add_parser(name, help=description, description=description)
        sub.set_defaults(func=func)
        sub.add_argument("--batch-size", type=int, default=16)
        return sub

    sub = command("directions", run_directions, "Find one destructive direction per layer.")
    sub.add_argument("--model", default=DEFAULT_MODEL)
    sub.add_argument("--out", type=Path, default=Path("runs/directions.pt"))

    sub = command("sweep", run_sweep, "Test each layer's direction with hooks, pick the best.")
    sub.add_argument("--directions", type=Path, default=Path("runs/directions.pt"))
    sub.add_argument("--layers", type=int_list, default=int_list("4-24"))
    sub.add_argument("--alphas", type=float_list, default=[1.0, 0.5])
    sub.add_argument("--min-benign", type=float, default=0.9)
    sub.add_argument("--max-new-tokens", type=int, default=160)
    sub.add_argument("--out", type=Path, default=Path("runs/sweep.json"))

    sub = command("export", run_export, "Bake one direction into the weights and save the model.")
    sub.add_argument("--directions", type=Path, default=Path("runs/directions.pt"))
    sub.add_argument("--layer", type=int, required=True)
    sub.add_argument("--alpha", type=float, default=1.0)
    sub.add_argument("--out", type=Path, required=True)

    sub = command("eval", run_eval, "Score a model on the evaluation sets.")
    sub.add_argument("--model", default=DEFAULT_MODEL)
    sub.add_argument("--max-new-tokens", type=int, default=160)
    sub.add_argument("--out", type=Path, default=Path("runs/eval.json"))

    sub = command("analyze", run_analyze, "Run controls, refusal overlap, and intervals.")
    sub.add_argument("--directions", type=Path, default=Path("runs/directions.pt"))
    sub.add_argument("--layer", type=int, default=20)
    sub.add_argument("--alpha", type=float, default=1.0)
    sub.add_argument("--controls", type=int, default=5)
    sub.add_argument("--seed", type=int, default=0)
    sub.add_argument("--max-new-tokens", type=int, default=160)
    sub.add_argument("--base-eval", type=Path, default=Path("runs/eval-base.json"))
    sub.add_argument("--edited-eval", type=Path, default=Path("runs/eval-edited.json"))
    sub.add_argument("--out", type=Path, default=Path("runs/analysis.json"))

    sub = command("plots", run_plots, "Draw the README charts from the saved results.")
    sub.add_argument("--sweep", type=Path, default=Path("runs/sweep.json"))
    sub.add_argument("--analysis", type=Path, default=Path("runs/analysis.json"))
    sub.add_argument("--out", type=Path, default=Path("docs/img"))

    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
