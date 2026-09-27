"""Training the surrogate (spec M1 to M4).

The path-gain head is trained with L1 loss in dB on cells where the ray tracer has power
(spec M2: "valid" means power above zero; PR 3 set no power threshold), and the power head
with binary cross-entropy on all cells (spec M1b). Only the train and validation splits are
loaded; the best epoch is chosen on validation. Synthetic terrain.
"""

import hashlib
import json
import platform
import resource
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from sionna_twin_ops.augment import SIN_CHANNEL, VARIANTS
from sionna_twin_ops.dataset import read_manifest
from sionna_twin_ops.features import INPUT_CHANNELS, map_inputs, targets, terrain_features
from sionna_twin_ops.model import RESIDUAL_SCALE_DB, UNet, parameter_count
from sionna_twin_ops.site import Site
from sionna_twin_ops.terrain import GRID_SPACING_M, generate_terrain

LOS_CHANNEL = 5  # features.py
# ASSUMPTION (as in `twin fold-check`): a cell whose traced gain exceeds B0 by this much is
# reflection-dominated. The residual target is traced minus B0, so the test is on it.
REFLECTION_EXCESS_DB = 3.0
POWER_LOSS_WEIGHT = 1.0  # ASSUMPTION: total = L1(dB) / RESIDUAL_SCALE_DB + weight * BCE
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4


@dataclass(frozen=True)
class SplitData:
    """One split held as tensors.

    Attributes:
        inputs: (N, channels, 128, 128).
        residual: (N, 128, 128) residual over B0 in dB, 0 where there is no power.
        power: (N, 128, 128) power mask.
        files: Map file names, in order.
    """

    inputs: Tensor
    residual: Tensor
    power: Tensor
    files: tuple[str, ...]


def load_split(dataset: Path, split: str, storage: type[np.floating[Any]]) -> SplitData:
    """Build the tensors of one split from a dataset directory.

    Args:
        dataset: Dataset directory (manifest.jsonl and maps/).
        split: "train", "validation" or "test".
        storage: Array dtype the split is held in (np.float16 or np.float32); batches are
            cast to float32 before use.

    Returns:
        The split.

    Raises:
        ValueError: If the manifest holds no map of that split.
    """
    lines = sorted(
        (line for line in read_manifest(dataset) if line["split"] == split),
        key=lambda line: line["file"],
    )
    if not lines:
        raise ValueError(f"no {split} maps in {dataset}")
    # Preallocated: stacking lists would briefly hold every map twice.
    n = len(lines)
    inputs = np.empty((n, INPUT_CHANNELS, 128, 128), dtype=storage)
    residuals = np.empty((n, 128, 128), dtype=storage)
    powers = np.empty((n, 128, 128), dtype=storage)
    # Maps are sorted by file name, so each terrain's maps are contiguous: only the current
    # terrain's features are held (each holds its profiles, about 67 MB).
    current_id, features = -1, None
    for index, line in enumerate(lines):
        if line["terrain_id"] != current_id:
            current_id = line["terrain_id"]
            terrain = generate_terrain(current_id, GRID_SPACING_M)
            features = terrain_features(terrain, Site(**line["site"]))
        if features is None:
            raise RuntimeError(f"no features built for {line['file']}")
        x, b0 = map_inputs(features, line["azimuth_deg"], line["tilt_deg"])
        residual, power = targets(np.load(dataset / "maps" / line["file"]), b0)
        inputs[index], residuals[index], powers[index] = x, residual, power
    return SplitData(
        inputs=torch.from_numpy(inputs),
        residual=torch.from_numpy(residuals),
        power=torch.from_numpy(powers),
        files=tuple(line["file"] for line in lines),
    )


def losses(output: Tensor, residual: Tensor, power: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """L1 in dB on cells with power, BCE on all cells, and the weighted total.

    Args:
        output: Model output (batch, 2, 128, 128).
        residual: Target residual in dB (batch, 128, 128).
        power: Power mask (batch, 128, 128).

    Returns:
        (L1 in dB, BCE, total).
    """
    predicted_db = output[:, 0] * RESIDUAL_SCALE_DB
    l1 = (torch.abs(predicted_db - residual) * power).sum() / power.sum().clamp(min=1.0)
    bce = nn.functional.binary_cross_entropy_with_logits(output[:, 1], power)
    return l1, bce, l1 / RESIDUAL_SCALE_DB + POWER_LOSS_WEIGHT * bce


def evaluate(model: nn.Module, data: SplitData, device: torch.device) -> dict[str, float]:
    """Validation metrics over a whole split.

    Args:
        model: The model.
        data: The split.
        device: Where to run.

    Returns:
        L1 in dB on cells with power (all; NLOS only; LOS split into direct- and
        reflection-dominated), BCE, total loss and power-mask accuracy.
    """
    model.eval()
    l1_sum = bce_sum = correct = cells = power_cells = 0.0
    nlos_sum = nlos_cells = 0.0
    direct_sum = direct_cells = reflected_sum = reflected_cells = 0.0
    with torch.no_grad():
        for start in range(0, len(data.files), BATCH_SIZE):
            x = data.inputs[start : start + BATCH_SIZE].to(device).float()
            r = data.residual[start : start + BATCH_SIZE].to(device).float()
            p = data.power[start : start + BATCH_SIZE].to(device).float()
            out = model(x)
            error = torch.abs(out[:, 0] * RESIDUAL_SCALE_DB - r) * p
            l1_sum += float(error.sum())
            power_cells += float(p.sum())
            nlos = 1.0 - x[:, LOS_CHANNEL]
            nlos_sum += float((error * nlos).sum())
            nlos_cells += float((p * nlos).sum())
            reflected = x[:, LOS_CHANNEL] * p * (r >= REFLECTION_EXCESS_DB).float()
            direct = x[:, LOS_CHANNEL] * p * (r < REFLECTION_EXCESS_DB).float()
            reflected_sum += float((error * reflected).sum())
            reflected_cells += float(reflected.sum())
            direct_sum += float((error * direct).sum())
            direct_cells += float(direct.sum())
            bce_sum += float(
                nn.functional.binary_cross_entropy_with_logits(out[:, 1], p, reduction="sum")
            )
            correct += float(((out[:, 1] > 0).float() == p).sum())
            cells += p.numel()
    l1 = l1_sum / power_cells
    bce = bce_sum / cells
    return {
        "l1_db": l1,
        "l1_nlos_db": nlos_sum / nlos_cells,
        "l1_los_direct_db": direct_sum / direct_cells,
        "l1_los_reflection_db": reflected_sum / reflected_cells,
        "bce": bce,
        "total": l1 / RESIDUAL_SCALE_DB + POWER_LOSS_WEIGHT * bce,
        "power_accuracy": correct / cells,
    }


def transform_batch(
    inputs: Tensor, rasters: list[Tensor], k: int, mirror: bool
) -> tuple[Tensor, list[Tensor]]:
    """The same variant on a training batch on any device.

    Args:
        inputs: (batch, channels, rows, columns).
        rasters: Other (batch, rows, columns) tensors, such as the targets.
        k: Clockwise quarter turns.
        mirror: Mirror east-west first.

    Returns:
        (transformed inputs, transformed rasters).
    """

    def apply(t: Tensor) -> Tensor:
        if mirror:
            t = torch.flip(t, dims=(-1,))
        return torch.rot90(t, k=k, dims=(-2, -1))

    x = apply(inputs)
    if mirror:
        x = x.clone()
        x[:, SIN_CHANNEL] = -x[:, SIN_CHANNEL]
    return x, [apply(r) for r in rasters]


def augment_batch(
    x: Tensor, r: Tensor, p: Tensor, generator: torch.Generator
) -> tuple[Tensor, Tensor, Tensor]:
    """Give every sample of a training batch its own random exact symmetry.

    Args:
        x: Inputs (batch, channels, rows, columns).
        r: Residual targets (batch, rows, columns).
        p: Power masks (batch, rows, columns).
        generator: Seeded source of the variant choices.

    Returns:
        The transformed (x, r, p).
    """
    picks = torch.randint(len(VARIANTS), (x.shape[0],), generator=generator).tolist()
    xs, rs, ps = [], [], []
    for i, pick in enumerate(picks):
        k, mirror = VARIANTS[pick]
        xi, (ri, pi) = transform_batch(x[i : i + 1], [r[i : i + 1], p[i : i + 1]], k, mirror)
        xs.append(xi)
        rs.append(ri)
        ps.append(pi)
    return torch.cat(xs), torch.cat(rs), torch.cat(ps)


def train(
    dataset: Path,
    seed: int,
    epochs: int,
    width: int,
    augmentation: str,
    device_name: str,
    out: Path,
) -> dict[str, Any]:
    """Train one seed and save the best-on-validation weights and the run record.

    Args:
        dataset: Dataset directory.
        seed: Training seed (initialisation and batch order).
        epochs: Training epochs.
        width: U-Net width at the first level.
        augmentation: "none", or "symmetry" for the 8 exact map symmetries (train only).
        device_name: "cuda" or "cpu"; chosen explicitly, never by fallback.
        out: Run directory for model.pt and meta.json.

    Returns:
        The run record written to meta.json.

    Raises:
        RuntimeError: If "cuda" is asked for and PyTorch sees no CUDA device.
        ValueError: If the augmentation is not "none" or "symmetry".
    """
    if augmentation not in ("none", "symmetry"):
        raise ValueError(f"unknown augmentation {augmentation!r}")
    from sionna_twin_ops.provenance import commit, gpu

    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was asked for but PyTorch sees no CUDA device")
    device = torch.device(device_name)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    started = time.perf_counter()
    # The training split is held in float16 to fit WSL's memory; validation stays float32,
    # so every reported number comes from float32 inputs and targets.
    train_data = load_split(dataset, "train", np.float16)
    val_data = load_split(dataset, "validation", np.float32)
    loaded = time.perf_counter() - started
    peak_rss_mib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0  # Linux: KiB
    load_record = {"loaded_s": round(loaded, 1), "peak_rss_mib": round(peak_rss_mib)}
    print(json.dumps(load_record), flush=True)

    model = UNet(width).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    steps = epochs * -(-len(train_data.files) // BATCH_SIZE)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)
    order = torch.Generator().manual_seed(seed)
    variants = torch.Generator().manual_seed(seed + 10_000)
    out.mkdir(parents=True, exist_ok=True)
    history = []
    best = None
    for epoch in range(1, epochs + 1):
        model.train()
        permutation = torch.randperm(len(train_data.files), generator=order)
        train_sum = 0.0
        for start in range(0, len(permutation), BATCH_SIZE):
            idx = permutation[start : start + BATCH_SIZE]
            x = train_data.inputs[idx].to(device).float()
            r = train_data.residual[idx].to(device).float()
            p = train_data.power[idx].to(device).float()
            if augmentation == "symmetry":
                x, r, p = augment_batch(x, r, p, variants)
            _, _, total = losses(model(x), r, p)
            optimizer.zero_grad()
            # Tensor.backward carries no type annotations in torch.
            total.backward()  # type: ignore[no-untyped-call]
            optimizer.step()
            schedule.step()
            train_sum += float(total.detach()) * len(idx)
        val = evaluate(model, val_data, device)
        history.append({"epoch": epoch, "train_total": train_sum / len(permutation), **val})
        if best is None or val["total"] < best["total"]:
            best = {"epoch": epoch, **val}
            torch.save(model.state_dict(), out / "model.pt")
        print(json.dumps(history[-1]), flush=True)

    manifest_sha = hashlib.sha256((dataset / "manifest.jsonl").read_bytes()).hexdigest()
    record = {
        "provenance": {
            "commit": commit(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": device_name,
            "gpu": gpu(),
        },
        "dataset": {"path": str(dataset.resolve()), "manifest_sha256": manifest_sha},
        "hyperparameters": {
            "seed": seed,
            "epochs": epochs,
            "width": width,
            "parameters": parameter_count(model),
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "power_loss_weight": POWER_LOSS_WEIGHT,
            "augmentation": augmentation,
            "schedule": "cosine over all steps",
            "optimizer": "AdamW",
            "cudnn_deterministic": True,
        },
        "maps": {"train": len(train_data.files), "validation": len(val_data.files)},
        "storage": {"train": "float16", "validation": "float32"},
        "peak_rss_mib_after_load": round(peak_rss_mib),
        "best": best,
        "history": history,
        "seconds": {"load": round(loaded, 1), "total": round(time.perf_counter() - started, 1)},
    }
    (out / "meta.json").write_text(json.dumps(record, indent=1), newline="\n")
    return record


def training_summary_data(runs: list[Path], context_runs: list[Path]) -> dict[str, Any]:
    """The committed record of the training runs: settings, best epochs, spread across seeds.

    Args:
        runs: Run directories, one per seed.
        context_runs: Earlier run directories quoted as context from their own records
            (not retrained, not recomputed); empty for none.

    Returns:
        JSON-safe data for `reports.training_summary_markdown`: the run records as written.

    Raises:
        ValueError: If the runs differ in anything but the seed.
    """
    metas = [json.loads((run / "meta.json").read_text()) for run in runs]
    shared = {k: v for k, v in metas[0]["hyperparameters"].items() if k != "seed"}
    for meta in metas[1:]:
        others = {k: v for k, v in meta["hyperparameters"].items() if k != "seed"}
        if others != shared or meta["dataset"] != metas[0]["dataset"]:
            raise ValueError("runs differ in settings or dataset, not only in seed")
    return {
        "kind": "training_summary",
        "reflection_excess_db": REFLECTION_EXCESS_DB,
        "runs": [str(run) for run in runs],
        "metas": metas,
        "context": [json.loads((run / "meta.json").read_text()) for run in context_runs],
    }
