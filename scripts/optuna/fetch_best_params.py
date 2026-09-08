#!/usr/bin/env python3
# Report the best trial of an Optuna study, ready to paste into a config.

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import optuna
import yaml
from omegaconf import OmegaConf


def quote(value: str) -> str:
    """Single-quote a value the shell would otherwise treat as a glob or a brace."""
    return f"'{value}'" if any(c in value for c in "[]{},") else value


def study_names(storage: str, wanted: str | None) -> list[str]:
    """Every study in the database, or just the one asked for."""
    names = [s.study_name for s in optuna.get_all_study_summaries(storage)]
    if wanted is None:
        return sorted(names)
    if wanted not in names:
        sys.exit(f"no study '{wanted}' in the database (have: {', '.join(sorted(names))})")
    return [wanted]


def as_yaml(params: dict[str, object]) -> str:
    """The parameters as a nested YAML block.

    Optuna parameter names are dotted hydra paths, so they are turned back into nesting; leaf lists
    come out in flow style ('[128, 64, 32]'), as the experiment configs write widths.
    """
    dotlist = [f"{k}={v}" for k, v in sorted(params.items())]
    nested = OmegaConf.to_container(OmegaConf.from_dotlist(dotlist), resolve=True)
    return yaml.dump(nested, default_flow_style=None, sort_keys=False).rstrip("\n")


def report(study: optuna.Study) -> None:
    """Print one study's best trial: value, hydra overrides, YAML block."""
    trial = study.best_trial
    n_done = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    print(
        f"# {study.study_name}: best of {n_done} complete trial(s) is trial {trial.number}, "
        f"val/acc_best={trial.value:.6f}"
    )
    print("  " + " ".join(f"{k}={quote(str(v))}" for k, v in sorted(trial.params.items())))
    print()
    for line in as_yaml(trial.params).splitlines():
        print(f"  {line}")
    print()


def _dump_csv(study: optuna.Study, name: str) -> None:
    """Every trial of one study as CSV on stdout, the study name as the first column."""
    frame = study.trials_dataframe()
    frame.insert(0, "study", name)
    frame.to_csv(sys.stdout, index=False)


def _report_one(study: optuna.Study, name: str, as_csv: bool) -> None:
    """Report one study, as its trials or as its best trial."""
    if as_csv:
        _dump_csv(study, name)
        return
    try:
        report(study)
    except ValueError:  # no completed trial yet
        print(f"# {name}: no completed trial yet\n")


def _parse_args() -> argparse.Namespace:
    """Command line: the database, the studies to report and the shape of the report."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--db", required=True, help="Path to the study database.")
    parser.add_argument("--study", help="Study name; omit together with --all for a listing.")
    parser.add_argument("--all", action="store_true", help="Report every study in the database.")
    parser.add_argument(
        "--csv",
        action="store_true",
        help="Write every trial of the selected studies as CSV to stdout.",
    )
    args = parser.parse_args()
    if not args.all and not args.study:
        sys.exit("pass --study <name> or --all")
    return args


def main() -> int:
    """Report every study the command line selects."""
    args = _parse_args()
    db = Path(args.db).expanduser()
    if not db.exists():
        sys.exit(f"no such database: {db}")
    storage = f"sqlite:///{db}"
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    for name in study_names(storage, None if args.all else args.study):
        _report_one(optuna.load_study(study_name=name, storage=storage), name, args.csv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
