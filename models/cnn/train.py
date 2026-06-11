#!/usr/bin/env python3
"""Train lightweight patch-CNN for inversion masks; save predictions and metrics."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.paths import CLOUDS_ROOT, MODELS_ARTIFACTS_DIR, add_era5_arg, add_input_arg
from models.inversion.cnn_dataset import PATCH_SIZE, build_inversion_cnn_dataset
from models.inversion.dataset import (
    SPLIT_NAMES,
    Split,
    build_agreement_map,
    masks_to_fraction,
    pick_best_test_day,
)

OCCURRENCE_THRESHOLD = 0.05
MASK_THRESHOLD = 0.5
BATCH_SIZE = 512
MAX_EPOCHS = 12
LEARNING_RATE = 1e-3
PATIENCE = 3


class PatchCNN(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x)).squeeze(-1)


def _split_mask(sample_split: np.ndarray, split: Split) -> np.ndarray:
    return sample_split == int(split)


def _hourly_split_mask(hourly_splits: np.ndarray, split: Split) -> np.ndarray:
    return hourly_splits == int(split)


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


def _predict_full_masks(
    model: PatchCNN,
    dataset,
    time_indices: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    n_lat, n_lon = dataset.z.shape
    n_sel = len(time_indices)
    predicted = np.full((n_sel, n_lat, n_lon), np.nan, dtype=np.float32)
    ridge = dataset.ridge_mask()
    model.eval()

    with torch.no_grad():
        for out_t, t in enumerate(time_indices):
            sel = dataset.sample_time_idx == t
            if not np.any(sel):
                continue
            xb = torch.from_numpy(dataset.X[sel]).to(device)
            preds = model(xb).cpu().numpy()
            preds = np.clip(preds, 0.0, 1.0)
            lat_idx = dataset.sample_lat_idx[sel]
            lon_idx = dataset.sample_lon_idx[sel]
            slab = np.full((n_lat, n_lon), np.nan, dtype=np.float32)
            slab[lat_idx, lon_idx] = preds.astype(np.float32)
            slab[~ridge] = np.nan
            predicted[out_t] = slab
    return predicted


def _train_model(
    model: PatchCNN,
    dataset,
    device: torch.device,
) -> PatchCNN:
    train_m = _split_mask(dataset.sample_split, Split.TRAIN)
    val_m = _split_mask(dataset.sample_split, Split.VAL)

    x_train = torch.from_numpy(dataset.X[train_m].copy())
    y_train = torch.from_numpy(dataset.y[train_m].copy())
    train_loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )
    print(f"  training on {x_train.shape[0]} patches...", flush=True)

    if np.any(val_m):
        x_val = torch.from_numpy(dataset.X[val_m]).to(device)
        y_val = dataset.y[val_m]
    else:
        x_val = None
        y_val = None

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_state = None
    best_val_mae = float("inf")
    stale_epochs = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
            n_batches += 1

        if x_val is not None and y_val is not None:
            model.eval()
            with torch.no_grad():
                val_pred = model(x_val).cpu().numpy()
            val_mae = mean_absolute_error(y_val, np.clip(val_pred, 0.0, 1.0))
            print(f"  epoch {epoch:2d}  train_loss={epoch_loss / max(n_batches, 1):.4f}  val MAE={val_mae:.4f}")
            if val_mae < best_val_mae:
                best_val_mae = val_mae
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= PATIENCE:
                    print(f"  early stop at epoch {epoch}")
                    break
        else:
            print(f"  epoch {epoch:2d}  train_loss={epoch_loss / max(n_batches, 1):.4f} (no val split)")
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


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
    print(f"Building CNN dataset for {region} (z_min={z_min_m} m, mode={inversion_mode})...")
    dataset = build_inversion_cnn_dataset(
        region=region,
        z_min_m=z_min_m,
        inversion_mode=inversion_mode,
        icon_nc=icon_nc,
        era5_grib=era5_grib,
        max_hours=max_hours,
    )
    print(
        f"  {dataset.n_timesteps} hours, {dataset.X.shape[0]} ridge patches, "
        f"{dataset.X.shape[1]} channels, patch {PATCH_SIZE}x{PATCH_SIZE}"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  device: {device}")
    model = PatchCNN(in_channels=dataset.X.shape[1]).to(device)
    model = _train_model(model, dataset, device)

    train_m = _split_mask(dataset.sample_split, Split.TRAIN)
    val_m = _split_mask(dataset.sample_split, Split.VAL)
    test_m = _split_mask(dataset.sample_split, Split.TEST)

    all_time_idx = np.arange(dataset.n_timesteps, dtype=np.int32)
    predicted_masks = _predict_full_masks(model, dataset, all_time_idx, device)
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

    hourly_csv = output_dir / f"{prefix}_cnn_hourly_predictions.csv"
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
        "model_type": "patch_cnn",
        "channel_names": dataset.channel_names,
        "channel_mean": dataset.channel_mean.tolist(),
        "channel_std": dataset.channel_std.tolist(),
        "patch_size": PATCH_SIZE,
        "occurrence_threshold": OCCURRENCE_THRESHOLD,
        "mask_threshold": MASK_THRESHOLD,
        "best_test_day_start": best_day_start,
        "best_test_day_frames": best_day_frames,
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
    }

    masks_npz = output_dir / f"{prefix}_cnn_predicted_masks.npz"
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

    model_path = output_dir / f"{prefix}_cnn.pt"
    torch.save({"state_dict": model.state_dict(), "meta": meta}, model_path)

    test_t_idx = np.where(_hourly_split_mask(dataset.hourly_splits, Split.TEST))[0]
    model.eval()
    with torch.no_grad():
        if np.any(test_m):
            test_pred = model(torch.from_numpy(dataset.X[test_m]).to(device)).cpu().numpy()
            test_metrics = _cell_metrics(dataset.y[test_m], test_pred)
            test_hourly = _hourly_metrics(
                dataset.hourly_actual_fraction[test_t_idx],
                predicted_fraction[test_t_idx],
            ) if test_t_idx.size else {"fraction_mae": float("nan"), "fraction_rmse": float("nan")}
        else:
            test_pred = model(torch.from_numpy(dataset.X[train_m]).to(device)).cpu().numpy()
            test_metrics = _cell_metrics(dataset.y[train_m], test_pred)
            test_hourly = {"fraction_mae": float("nan"), "fraction_rmse": float("nan")}
            test_t_idx = np.where(_hourly_split_mask(dataset.hourly_splits, Split.TRAIN))[0]

    report_dir = CLOUDS_ROOT / "reports" / "phase04"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "cnn_training_report.md"
    lines = [
        "# Patch-CNN inversion — training report",
        "",
        f"- **Generated:** {meta['generated_utc']}",
        f"- **Region:** `{region}`",
        f"- **Inversion mode:** `{inversion_mode}`",
        f"- **Ridge cutoff:** `{z_min_m}` m",
        f"- **Model:** `{model_path.relative_to(CLOUDS_ROOT)}`",
        f"- **Hourly CSV:** `{hourly_csv.relative_to(CLOUDS_ROOT)}`",
        f"- **Masks NPZ:** `{masks_npz.relative_to(CLOUDS_ROOT)}`",
        "",
        "## Dataset",
        "",
        f"- Timesteps: **{dataset.n_timesteps}**",
        f"- Ridge patches: **{dataset.X.shape[0]}**",
        f"- Channels: **{len(dataset.channel_names)}** ({', '.join(dataset.channel_names)})",
        f"- Patch size: **{PATCH_SIZE}×{PATCH_SIZE}**",
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
        "- Hourly occurrence uses ridge inversion fraction ≥ 0.05.",
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
    parser.add_argument("--region", default="west", choices=("west", "east", "east_core"))
    parser.add_argument("--z-min-m", type=float, default=800.0)
    parser.add_argument(
        "--inversion-mode",
        choices=("deck_only", "phenomenological"),
        default="deck_only",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=MODELS_ARTIFACTS_DIR,
    )
    parser.add_argument("--max-hours", type=int, default=None, help="Limit hours for quick tests")
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
