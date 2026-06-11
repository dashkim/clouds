#!/usr/bin/env python3
"""Train Ridge regression for inversion masks; save predictions and metrics."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.paths import CLOUDS_ROOT, MODELS_ARTIFACTS_DIR, add_era5_arg, add_input_arg
from models.inversion.dataset import (
    SPLIT_NAMES,
    Split,
    build_agreement_map,
    build_inversion_ml_dataset,
    masks_to_fraction,
    pick_best_test_day,
)

ALPHAS = (0.1, 1.0, 10.0)
OCCURRENCE_THRESHOLD = 0.05
MASK_THRESHOLD = 0.5


def _split_mask(sample_split: np.ndarray, split: Split) -> np.ndarray:
    return sample_split == int(split)


def _hourly_split_mask(hourly_splits: np.ndarray, split: Split) -> np.ndarray:
    return hourly_splits == int(split)


def _predict_full_masks(
    model: Pipeline,
    dataset,
    time_indices: np.ndarray,
) -> np.ndarray:
    """Predict inversion mask for selected timesteps."""
    n_lat, n_lon = dataset.z.shape
    n_sel = len(time_indices)
    predicted = np.full((n_sel, n_lat, n_lon), np.nan, dtype=np.float32)
    ridge = dataset.ridge_mask()

    for out_t, t in enumerate(time_indices):
        sel = dataset.sample_time_idx == t
        if not np.any(sel):
            continue
        preds = model.predict(dataset.X[sel])
        preds = np.clip(preds, 0.0, 1.0)
        lat_idx = dataset.sample_lat_idx[sel]
        lon_idx = dataset.sample_lon_idx[sel]
        slab = np.full((n_lat, n_lon), np.nan, dtype=np.float32)
        slab[lat_idx, lon_idx] = preds.astype(np.float32)
        slab[~ridge] = np.nan
        predicted[out_t] = slab
    return predicted


def _cell_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_bin = (y_true >= MASK_THRESHOLD).astype(int)
    p_bin = (np.clip(y_pred, 0.0, 1.0) >= MASK_THRESHOLD).astype(int)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "precision": float(precision_score(y_bin, p_bin, zero_division=0)),
        "recall": float(recall_score(y_bin, p_bin, zero_division=0)),
        "f1": float(f1_score(y_bin, p_bin, zero_division=0)),
    }


def _hourly_metrics(
    actual_fraction: np.ndarray,
    predicted_fraction: np.ndarray,
) -> dict[str, float]:
    return {
        "fraction_mae": float(mean_absolute_error(actual_fraction, predicted_fraction)),
        "fraction_rmse": float(np.sqrt(mean_squared_error(actual_fraction, predicted_fraction))),
    }


def train_and_save(
    *,
    region: str,
    z_min_m: float,
    inversion_mode: str,
    output_dir: Path,
    icon_nc: Path | None,
    era5_grib: Path | None,
    max_hours: int | None,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = region
    print(f"Building dataset for {region} (z_min={z_min_m} m, mode={inversion_mode})...")
    dataset = build_inversion_ml_dataset(
        region=region,
        z_min_m=z_min_m,
        inversion_mode=inversion_mode,
        icon_nc=icon_nc,
        era5_grib=era5_grib,
        max_hours=max_hours,
    )
    print(
        f"  {dataset.n_timesteps} hours, {dataset.X.shape[0]} ridge samples, "
        f"{dataset.X.shape[1]} features"
    )

    train_m = _split_mask(dataset.sample_split, Split.TRAIN)
    val_m = _split_mask(dataset.sample_split, Split.VAL)
    test_m = _split_mask(dataset.sample_split, Split.TEST)

    if not np.any(train_m):
        raise ValueError("No training samples — widen time range or check split logic")

    best_alpha = ALPHAS[0]
    best_val_mae = float("inf")
    best_model: Pipeline | None = None

    for alpha in ALPHAS:
        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=alpha)),
        ])
        pipe.fit(dataset.X[train_m], dataset.y[train_m])
        if np.any(val_m):
            val_pred = np.clip(pipe.predict(dataset.X[val_m]), 0.0, 1.0)
            val_mae = mean_absolute_error(dataset.y[val_m], val_pred)
            print(f"  alpha={alpha:g}  val MAE={val_mae:.4f}")
            if val_mae < best_val_mae:
                best_val_mae = val_mae
                best_alpha = alpha
                best_model = pipe
        else:
            train_pred = np.clip(pipe.predict(dataset.X[train_m]), 0.0, 1.0)
            train_mae = mean_absolute_error(dataset.y[train_m], train_pred)
            print(f"  alpha={alpha:g}  train MAE={train_mae:.4f} (no val split in range)")
            if train_mae < best_val_mae:
                best_val_mae = train_mae
                best_alpha = alpha
                best_model = pipe

    assert best_model is not None
    print(f"Selected alpha={best_alpha:g}")

    all_time_idx = np.arange(dataset.n_timesteps, dtype=np.int32)
    predicted_masks = _predict_full_masks(best_model, dataset, all_time_idx)
    predicted_fraction = masks_to_fraction(predicted_masks, dataset.z, z_min_m)

    hourly_rows: list[dict] = []
    for t in range(dataset.n_timesteps):
        split_name = SPLIT_NAMES[int(dataset.hourly_splits[t])]
        act_frac = float(dataset.hourly_actual_fraction[t])
        pred_frac = float(predicted_fraction[t])
        hourly_rows.append({
            "region": region,
            "time": str(pd.Timestamp(dataset.times[t])),
            "split": split_name,
            "actual_fraction": act_frac,
            "predicted_fraction": pred_frac,
            "actual_occurrence": act_frac >= OCCURRENCE_THRESHOLD,
            "predicted_occurrence": pred_frac >= OCCURRENCE_THRESHOLD,
        })

    hourly_csv = output_dir / f"{prefix}_hourly_predictions.csv"
    pd.DataFrame(hourly_rows).to_csv(hourly_csv, index=False)

    agreement = np.empty_like(predicted_masks)
    for t in range(dataset.n_timesteps):
        agreement[t] = build_agreement_map(
            dataset.actual_masks[t],
            predicted_masks[t],
            z=dataset.z,
            z_min_m=z_min_m,
        )

    best_day_start, best_day_frames = pick_best_test_day(
        dataset.times,
        dataset.hourly_actual_fraction,
        dataset.hourly_splits,
    )

    meta = {
        "region": region,
        "z_min_m": z_min_m,
        "inversion_mode": inversion_mode,
        "inversion_params": dataset.inversion_params.to_dict(),
        "alpha": best_alpha,
        "feature_names": dataset.feature_names,
        "occurrence_threshold": OCCURRENCE_THRESHOLD,
        "mask_threshold": MASK_THRESHOLD,
        "best_test_day_start": best_day_start,
        "best_test_day_frames": best_day_frames,
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
    }

    masks_npz = output_dir / f"{prefix}_predicted_masks.npz"
    np.savez_compressed(
        masks_npz,
        times=np.asarray([str(pd.Timestamp(t)) for t in dataset.times]),
        actual=dataset.actual_masks,
        predicted=predicted_masks,
        agreement=agreement,
        lon=dataset.lon,
        lat=dataset.lat,
        z=dataset.z,
        meta_json=np.array([json.dumps(meta)]),
    )

    model_path = output_dir / f"{prefix}_ridge.joblib"
    joblib.dump({"pipeline": best_model, "meta": meta}, model_path)

    test_t_idx = np.where(_hourly_split_mask(dataset.hourly_splits, Split.TEST))[0]
    test_cell_m = test_m
    if np.any(test_cell_m):
        test_metrics = _cell_metrics(
            dataset.y[test_cell_m],
            best_model.predict(dataset.X[test_cell_m]),
        )
        test_hourly = _hourly_metrics(
            dataset.hourly_actual_fraction[test_t_idx],
            predicted_fraction[test_t_idx],
        ) if test_t_idx.size else {"fraction_mae": float("nan"), "fraction_rmse": float("nan")}
    else:
        test_metrics = _cell_metrics(dataset.y[train_m], best_model.predict(dataset.X[train_m]))
        test_hourly = {"fraction_mae": float("nan"), "fraction_rmse": float("nan")}
        test_t_idx = np.where(_hourly_split_mask(dataset.hourly_splits, Split.TRAIN))[0]

    report_dir = CLOUDS_ROOT / "reports" / "phase04"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"{region}_ridge_training_report.md"
    lines = [
        "# Ridge inversion regression — training report",
        "",
        f"- **Generated:** {meta['generated_utc']}",
        f"- **Region:** `{region}`",
        f"- **Inversion mode:** `{inversion_mode}`",
        f"- **Ridge cutoff:** `{z_min_m}` m",
        f"- **Alpha:** `{best_alpha}`",
        f"- **Model:** `{model_path.relative_to(CLOUDS_ROOT)}`",
        f"- **Hourly CSV:** `{hourly_csv.relative_to(CLOUDS_ROOT)}`",
        f"- **Masks NPZ:** `{masks_npz.relative_to(CLOUDS_ROOT)}`",
        "",
        "## Dataset",
        "",
        f"- Timesteps: **{dataset.n_timesteps}**",
        f"- Ridge samples: **{dataset.X.shape[0]}**",
        f"- Train / val / test samples: "
        f"**{train_m.sum()}** / **{val_m.sum()}** / **{test_m.sum()}**",
        "",
        "## Test metrics (ridge cells)",
        "",
        f"- MAE: **{test_metrics['mae']:.4f}**",
        f"- RMSE: **{test_metrics['rmse']:.4f}**",
        f"- Precision @ 0.5: **{test_metrics['precision']:.3f}**",
        f"- Recall @ 0.5: **{test_metrics['recall']:.3f}**",
        f"- F1 @ 0.5: **{test_metrics['f1']:.3f}**",
        "",
        "## Test hourly fraction",
        "",
        f"- Fraction MAE: **{test_hourly['fraction_mae']:.4f}**",
        f"- Fraction RMSE: **{test_hourly['fraction_rmse']:.4f}**",
        "",
        "## Best test comparison day",
        "",
        f"- Start: **`{best_day_start}`**",
        f"- Frames: **{best_day_frames}**",
        "",
        "Render:",
        "",
        "```bash",
        "python models/render/compare.py \\",
        f"  --region {region} \\",
        f"  --predictions {masks_npz.relative_to(CLOUDS_ROOT)} \\",
        f"  --era5-start {best_day_start} \\",
        f"  --era5-frames {best_day_frames}",
        "```",
        "",
        "## Notes",
        "",
        "- Labels are rule-based (`detect_inversion_mask`) from ERA5 + ICON terrain.",
        "- Hourly occurrence uses inversion fraction ≥ 0.05.",
        "- Agreement map: 1=TP (green), 2=FP (red), 3=FN (blue).",
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"Wrote {model_path}")
    print(f"Wrote {hourly_csv}")
    print(f"Wrote {masks_npz}")
    print(f"Wrote {report_path}")
    print(f"Best test day: {best_day_start} ({best_day_frames} frames)")

    return {
        "meta": meta,
        "test_metrics": test_metrics,
        "test_hourly": test_hourly,
        "model_path": model_path,
        "hourly_csv": hourly_csv,
        "masks_npz": masks_npz,
        "report_path": report_path,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_input_arg(parser, kind="lonlat")
    add_era5_arg(parser)
    parser.add_argument("--region", default="east_core", choices=("west", "east", "east_core"))
    parser.add_argument("--z-min-m", type=float, default=800.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=MODELS_ARTIFACTS_DIR,
    )
    parser.add_argument("--max-hours", type=int, default=None, help="Limit hours for quick tests")
    parser.add_argument(
        "--inversion-mode",
        choices=("deck_only", "phenomenological"),
        default="phenomenological",
    )
    args = parser.parse_args()

    train_and_save(
        region=args.region,
        z_min_m=args.z_min_m,
        inversion_mode=args.inversion_mode,
        output_dir=args.output_dir,
        icon_nc=Path(args.input) if args.input else None,
        era5_grib=Path(args.era5_grib) if args.era5_grib else None,
        max_hours=args.max_hours,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
