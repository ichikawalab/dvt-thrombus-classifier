"""Discrimination within anatomical subgroups, from untouched held-out predictions.

Requires a local manifest that carries a subgroup column (for example the
venous segment of each image), so it runs only at the originating site::

    uv run python scripts\\within_segment_auc.py ^
        --runs outputs\\runs ^
        --input-csv data\\input.csv --segment-column segment_raw ^
        --group femoral=CFV,SFV --group popliteal=POP ^
        --group femoral_popliteal=CFV,SFV,POP ^
        --output outputs\\stats\\within_segment

Each ``--group NAME=value,value`` restricts the held-out predictions to the
images whose subgroup value is in the list; a parenthesised suffix such as
``"(visual read)"`` is ignored when matching. AUCs are computed exactly as in
the main report, per repetition on the restricted out-of-fold predictions, and
summarised with a patient-cluster bootstrap interval. Groups lacking either
class are reported with counts only.

Outputs ``segment_distribution.csv`` (images per subgroup value and label) and
``within_segment_auc.csv``.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from dvt_thrombus.analysis import (
    bootstrap_auc_ci,
    build_selected_predictions,
    fixed_model_selection,
    select_models,
)
from dvt_thrombus.metrics import roc_auc
from dvt_thrombus.report import ALL6, TOP1, TOP3, _load_run_artifacts, _validate_models

_SUFFIX_RE = re.compile(r"\s*\(.*\)\s*$")


def _parse_group(text: str) -> tuple[str, list[str]]:
    name, _, values = text.partition("=")
    members = [v.strip() for v in values.split(",") if v.strip()]
    if not name.strip() or not members:
        raise argparse.ArgumentTypeError(f"expected NAME=value[,value...], got {text!r}")
    return name.strip(), members


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--segment-column", default="segment")
    parser.add_argument("--group", type=_parse_group, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=5000)
    args = parser.parse_args(argv)

    sets, validation_auc, validation_predictions = _load_run_artifacts(args.runs)
    models = _validate_models(sets, validation_predictions)
    reference = sets[models[0]]
    keys = {
        (repetition, int(fold))
        for repetition in range(1, reference.n_repetitions + 1)
        for fold in np.unique(reference.frame[f"fold_{repetition}"])
    }
    strategies = {
        **{model: fixed_model_selection(keys, (model,)) for model in models},
        TOP1: select_models(validation_auc, k=1),
        TOP3: select_models(validation_auc, k=3),
        ALL6: fixed_model_selection(keys, models),
    }
    matrices = {name: build_selected_predictions(sets, sel) for name, sel in strategies.items()}

    manifest = pd.read_csv(args.input_csv, encoding="utf-8-sig", dtype=str)
    if args.segment_column not in manifest.columns:
        raise KeyError(f"{args.input_csv} has no column {args.segment_column!r}")
    manifest = manifest.set_index("image_id")
    missing = set(reference.image_id) - set(manifest.index)
    if missing:
        raise KeyError(f"{len(missing)} predicted image(s) are absent from the manifest")
    raw = manifest.loc[reference.image_id, args.segment_column].astype(str)
    segment = raw.str.replace(_SUFFIX_RE, "", regex=True).to_numpy()
    y_true, patient_id = reference.y_true, reference.patient_id

    args.output.mkdir(parents=True, exist_ok=True)
    distribution = (
        pd.crosstab(pd.Series(segment, name="segment"), pd.Series(y_true, name="label"))
        .rename(columns={0: "negative", 1: "positive"})
        .reset_index()
    )
    distribution["total"] = distribution["negative"] + distribution["positive"]
    distribution.to_csv(args.output / "segment_distribution.csv", index=False)

    groups = list(args.group) or [(value, [value]) for value in sorted(set(segment))]
    rows: list[dict[str, object]] = []
    for name, members in groups:
        mask = np.isin(segment, members)
        n_neg, n_pos = int((y_true[mask] == 0).sum()), int((y_true[mask] == 1).sum())
        base = {
            "group": name,
            "members": ", ".join(members),
            "n_images": int(mask.sum()),
            "n_negative": n_neg,
            "n_positive": n_pos,
            "n_patients": int(np.unique(patient_id[mask]).size),
        }
        if n_neg == 0 or n_pos == 0:
            rows.append({**base, "Model": None, "note": "only one class present"})
            continue
        for strategy, matrix in matrices.items():
            sub = matrix[mask]
            per_rep = [roc_auc(y_true[mask], sub[:, k]) for k in range(sub.shape[1])]
            ci = bootstrap_auc_ci(
                y_true[mask], sub, patient_id[mask], n_resamples=args.bootstrap_resamples
            )
            rows.append(
                {
                    **base,
                    "Model": strategy,
                    "AUC_mean": float(np.mean(per_rep)),
                    "AUC_sd": float(np.std(per_rep, ddof=1)),
                    "AUC_min": float(np.min(per_rep)),
                    "AUC_max": float(np.max(per_rep)),
                    "AUC_ci_lower": ci["ci_lower"],
                    "AUC_ci_upper": ci["ci_upper"],
                    "n_resamples": ci["n_resamples"],
                }
            )
    table = pd.DataFrame(rows)
    table.to_csv(args.output / "within_segment_auc.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(distribution.to_string(index=False))
        print()
        print(table.round(3).to_string(index=False))
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
