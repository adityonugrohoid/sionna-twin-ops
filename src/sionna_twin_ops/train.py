"""Training the surrogate (spec M1 to M4).

The path-gain head is trained with L1 loss in dB on cells where the ray tracer has power
(spec M2: "valid" means power above zero; PR 3 set no power threshold), and the power head
with binary cross-entropy on all cells (spec M1b). Only the train and validation splits are
loaded; the best epoch is chosen on validation. Synthetic terrain.
"""

import hashlib
import json
import platform
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from sionna_twin_ops.dataset import read_manifest
from sionna_twin_ops.features import map_inputs, targets, terrain_features
from sionna_twin_ops.model import RESIDUAL_SCALE_DB, UNet, parameter_count
from sionna_twin_ops.site import Site

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


def load_split(dataset: Path, split: str) -> SplitData:
    """Build the tensors of one split from a dataset directory.

    Args:
        dataset: Dataset directory (manifest.jsonl and maps/).
        split: "train", "validation" or "test".

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
    cache: dict[int, Any] = {}
    inputs, residuals, powers = [], [], []
    for line in lines:
        terrain_id = line["terrain_id"]
        if terrain_id not in cache:
            cache[terrain_id] = terrain_features(terrain_id, Site(**line["site"]))
        x, b0 = map_inputs(cache[terrain_id], line["azimuth_deg"], line["tilt_deg"])
        residual, power = targets(np.load(dataset / "maps" / line["file"]), b0)
        inputs.append(x)
        residuals.append(residual)
        powers.append(power)
    return SplitData(
        inputs=torch.from_numpy(np.stack(inputs)),
        residual=torch.from_numpy(np.stack(residuals)),
        power=torch.from_numpy(np.stack(powers)),
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
        L1 in dB on cells with power, BCE, total loss and power-mask accuracy.
    """
    model.eval()
    l1_sum = bce_sum = correct = cells = power_cells = 0.0
    with torch.no_grad():
        for start in range(0, len(data.files), BATCH_SIZE):
            x = data.inputs[start : start + BATCH_SIZE].to(device)
            r = data.residual[start : start + BATCH_SIZE].to(device)
            p = data.power[start : start + BATCH_SIZE].to(device)
            out = model(x)
            l1_sum += float((torch.abs(out[:, 0] * RESIDUAL_SCALE_DB - r) * p).sum())
            power_cells += float(p.sum())
            bce_sum += float(
                nn.functional.binary_cross_entropy_with_logits(out[:, 1], p, reduction="sum")
            )
            correct += float(((out[:, 1] > 0).float() == p).sum())
            cells += p.numel()
    l1 = l1_sum / power_cells
    bce = bce_sum / cells
    return {
        "l1_db": l1,
        "bce": bce,
        "total": l1 / RESIDUAL_SCALE_DB + POWER_LOSS_WEIGHT * bce,
        "power_accuracy": correct / cells,
    }


def train(
    dataset: Path, seed: int, epochs: int, width: int, device_name: str, out: Path
) -> dict[str, Any]:
    """Train one seed and save the best-on-validation weights and the run record.

    Args:
        dataset: Dataset directory.
        seed: Training seed (initialisation and batch order).
        epochs: Training epochs.
        width: U-Net width at the first level.
        device_name: "cuda" or "cpu"; chosen explicitly, never by fallback.
        out: Run directory for model.pt and meta.json.

    Returns:
        The run record written to meta.json.

    Raises:
        RuntimeError: If "cuda" is asked for and PyTorch sees no CUDA device.
    """
    from sionna_twin_ops.provenance import commit, gpu

    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was asked for but PyTorch sees no CUDA device")
    device = torch.device(device_name)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    started = time.perf_counter()
    train_data = load_split(dataset, "train")
    val_data = load_split(dataset, "validation")
    loaded = time.perf_counter() - started

    model = UNet(width).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    steps = epochs * -(-len(train_data.files) // BATCH_SIZE)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)
    order = torch.Generator().manual_seed(seed)
    out.mkdir(parents=True, exist_ok=True)
    history = []
    best = None
    for epoch in range(1, epochs + 1):
        model.train()
        permutation = torch.randperm(len(train_data.files), generator=order)
        train_sum = 0.0
        for start in range(0, len(permutation), BATCH_SIZE):
            idx = permutation[start : start + BATCH_SIZE]
            x = train_data.inputs[idx].to(device)
            r = train_data.residual[idx].to(device)
            p = train_data.power[idx].to(device)
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
            "schedule": "cosine over all steps",
            "optimizer": "AdamW",
            "cudnn_deterministic": True,
        },
        "maps": {"train": len(train_data.files), "validation": len(val_data.files)},
        "best": best,
        "history": history,
        "seconds": {"load": round(loaded, 1), "total": round(time.perf_counter() - started, 1)},
    }
    (out / "meta.json").write_text(json.dumps(record, indent=1), newline="\n")
    return record


def training_summary_markdown(runs: list[Path]) -> str:
    """The committed record of the training runs: settings, best epochs, spread across seeds.

    Args:
        runs: Run directories, one per seed.

    Returns:
        Markdown text.

    Raises:
        ValueError: If the runs differ in anything but the seed.
    """
    metas = [json.loads((run / "meta.json").read_text()) for run in runs]
    shared = {k: v for k, v in metas[0]["hyperparameters"].items() if k != "seed"}
    for meta in metas[1:]:
        others = {k: v for k, v in meta["hyperparameters"].items() if k != "seed"}
        if others != shared or meta["dataset"] != metas[0]["dataset"]:
            raise ValueError("runs differ in settings or dataset, not only in seed")
    best = [meta["best"] for meta in metas]

    def spread(key: str) -> str:
        values = np.array([b[key] for b in best])
        return f"{values.mean():.3f} (min {values.min():.3f}, max {values.max():.3f})"

    lines = [
        "# Training summary",
        "",
        "Synthetic terrain. The surrogate is a U-Net predicting the ray-traced path gain as a "
        "residual over B0, plus a logit for whether the ray tracer has power in each cell "
        "(spec M1, M1b). Trained on the train split, best epoch chosen on validation; the "
        "test split is not touched here. Written by `twin training-summary`.",
        "",
        "| setting | value |",
        "|---|---|",
        *(f"| {k} | {v} |" for k, v in shared.items()),
        f"| dataset | {metas[0]['dataset']['path']} (manifest sha256 "
        f"{metas[0]['dataset']['manifest_sha256'][:16]}...) |",
        f"| maps | train {metas[0]['maps']['train']}, "
        f"validation {metas[0]['maps']['validation']} |",
        "",
        "| provenance | values seen |",
        "|---|---|",
        *(
            f"| {k} | {'; '.join(sorted({str(m['provenance'][k]) for m in metas}))} |"
            for k in metas[0]["provenance"]
        ),
        "",
        "## Best epoch per seed (validation)",
        "",
        "L1 is the mean absolute error in dB over validation cells where the ray tracer has "
        "power; power accuracy is the share of all validation cells whose power logit has "
        "the right sign.",
        "",
        "| seed | best epoch | L1 (dB) | BCE | power accuracy | total loss | seconds |",
        "|---|---|---|---|---|---|---|",
    ]
    for meta in metas:
        b = meta["best"]
        lines.append(
            f"| {meta['hyperparameters']['seed']} | {b['epoch']} | {b['l1_db']:.3f} | "
            f"{b['bce']:.4f} | {b['power_accuracy'] * 100:.2f}% | {b['total']:.4f} | "
            f"{meta['seconds']['total']:.0f} |"
        )
    lines += [
        "",
        f"Across seeds: L1 {spread('l1_db')} dB; power accuracy {spread('power_accuracy')}; "
        f"BCE {spread('bce')}.",
    ]
    return "\n".join(lines) + "\n"
