"""The 3D viewer (spec E7): one self-contained HTML page that drapes maps over their terrain.

Data per case (terrain, azimuth, tilt): the ray-traced path gain, the surrogate's predicted
path gain and its power head. Maps are little-endian int16 in 0.1 dB steps (NO_POWER where
the ray tracer has none), the power head a bitmask, each zlib-compressed and base64-encoded;
the page inflates them with the browser's DecompressionStream. Each terrain's ground heights
are stored once. Plotly is loaded from cdn.jsdelivr.net at a pinned version. Synthetic terrain.
"""

import base64
import json
import zlib
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from sionna_twin_ops.dataset import AZIMUTHS_DEG, read_manifest
from sionna_twin_ops.features import map_inputs, targets, terrain_features
from sionna_twin_ops.site import SITE_CLASSES, SURFACE_HEIGHT_M, Site
from sionna_twin_ops.terrain import GRID_SPACING_M, generate_terrain

PLOTLY_URL = "https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.2/plotly.min.js"
NO_POWER = -32768  # int16 sentinel: the ray tracer has no power in the cell
BEAM_LENGTH_M = 2000.0


def pack_int16(values: NDArray[np.floating[Any]]) -> str:
    """Compressed base64 of little-endian int16 in tenths (NaN becomes NO_POWER).

    Args:
        values: Array to pack.

    Returns:
        Base64 text of the zlib stream.

    Raises:
        ValueError: If a value does not fit int16 at 0.1 resolution.
    """
    scaled = np.round(np.asarray(values, dtype=np.float64) * 10.0)
    finite = np.isfinite(scaled)
    if (np.abs(scaled[finite]) > 32767).any():
        raise ValueError("value outside the int16 range at 0.1 resolution")
    out = np.where(finite, scaled, NO_POWER).astype("<i2")
    return base64.b64encode(zlib.compress(out.tobytes(), 9)).decode("ascii")


def pack_bits(mask: NDArray[np.bool_]) -> str:
    """Compressed base64 of a boolean array, 8 cells per byte, first cell in the high bit.

    Args:
        mask: Boolean array.

    Returns:
        Base64 text of the zlib stream.
    """
    return base64.b64encode(zlib.compress(np.packbits(mask.ravel()).tobytes(), 9)).decode("ascii")


def viewer_cases(
    dataset: Path, split: str, azimuth_deg: float, tilt_deg: float
) -> list[tuple[int, float, float]]:
    """Every terrain of the split at one configuration, then the evaluation figure's terrains
    (the lowest id of each site class) at the other grid azimuths.

    Args:
        dataset: Dataset directory.
        split: Split the cases come from.
        azimuth_deg: Configuration shown for every terrain.
        tilt_deg: Tilt of every case.

    Returns:
        (terrain id, azimuth, tilt) per case.
    """
    lines = [line for line in read_manifest(dataset) if line["split"] == split]
    classes = {line["terrain_id"]: line["site"]["site_class"] for line in lines}
    terrains = sorted(classes)
    figure_terrains = [min(t for t in terrains if classes[t] == c) for c in SITE_CLASSES]
    cases = [(t, azimuth_deg, tilt_deg) for t in terrains]
    cases += [(t, a, tilt_deg) for t in figure_terrains for a in AZIMUTHS_DEG if a != azimuth_deg]
    return cases


def viewer_data(
    dataset: Path, split: str, run: Path, cases: list[tuple[int, float, float]]
) -> dict[str, Any]:
    """Everything the page draws, packed.

    Args:
        dataset: Dataset directory.
        split: Split the cases come from.
        run: Training run whose model predicts (its best checkpoint).
        cases: (terrain id, azimuth, tilt) per case; each must be in the split.

    Returns:
        {"terrains", "cases", "seed", "ray_tracing"} ready for JSON; "ray_tracing" lists the
        distinct solver settings of the shown maps.

    Raises:
        ValueError: If a case is not in the split.
    """
    import torch

    from sionna_twin_ops.evaluate import load_models
    from sionna_twin_ops.model import RESIDUAL_SCALE_DB

    lines = {
        (line["terrain_id"], line["azimuth_deg"], line["tilt_deg"]): line
        for line in read_manifest(dataset)
        if line["split"] == split
    }
    ((seed, model),) = load_models([run], torch.device("cpu"))
    terrains: dict[str, Any] = {}
    features_by_terrain: dict[int, Any] = {}
    packed_cases = []
    ray_tracing = set()
    for terrain_id, azimuth, tilt in cases:
        line = lines.get((terrain_id, azimuth, tilt))
        if line is None:
            raise ValueError(f"terrain {terrain_id}, azimuth {azimuth}, tilt {tilt} not in {split}")
        settings = line["settings"]
        ray_tracing.add(
            f"{settings['variant']}, {settings['samples_per_tx']:.0e} rays, max depth "
            f"{settings['max_depth']}, Sionna RT {line['provenance']['sionna-rt']}"
        )
        site = Site(**line["site"])
        if terrain_id not in features_by_terrain:
            terrain = generate_terrain(terrain_id, GRID_SPACING_M)
            features = terrain_features(terrain, site)
            features_by_terrain[terrain_id] = features
            terrains[str(terrain_id)] = {
                "ground": pack_int16(features.geometry.rx_m - SURFACE_HEIGHT_M),
                "site_class": site.site_class,
                "relief_m": round(terrain.params.relief_m),
                "site_ground_m": round(site.ground_m, 1),
                "antenna_m": round(site.antenna_m, 1),
                "percentile": round(site.percentile),
            }
        features = features_by_terrain[terrain_id]
        x, b0 = map_inputs(features, azimuth, tilt)
        residual, power_f = targets(np.load(dataset / "maps" / line["file"]), b0)
        truth = power_f > 0.5
        with torch.no_grad():
            out = model(torch.from_numpy(x[None])).numpy()[0]
        says_power = out[1] > 0
        packed_cases.append(
            {
                "terrain": terrain_id,
                "azimuth": azimuth,
                "tilt": tilt,
                "traced": pack_int16(np.where(truth, residual + b0, np.nan)),
                "predicted": pack_int16(b0 + out[0] * RESIDUAL_SCALE_DB),
                "says_power": pack_bits(says_power),
                "false_power": int((says_power & ~truth).sum()),
                "missed_power": int((~says_power & truth).sum()),
            }
        )
    return {
        "terrains": terrains,
        "cases": packed_cases,
        "seed": seed,
        "ray_tracing": sorted(ray_tracing),
    }


PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Surrogate map viewer</title>
<script src="__PLOTLY__"></script>
<style>
  body { margin: 0; background: #111; color: #ddd; font: 14px system-ui, sans-serif; }
  header { padding: 8px 16px; display: flex; flex-wrap: wrap; gap: 12px; align-items: center; }
  select, button { background: #222; color: #ddd; border: 1px solid #444; padding: 4px 8px;
    font: inherit; }
  button.on { border-color: #e0a030; color: #e0a030; }
  #info { color: #aaa; }
  #plot { width: 100vw; height: calc(100vh - 110px); }
  footer { padding: 6px 16px; color: #999; font-size: 12px; }
</style>
</head>
<body>
<header>
  <label>Case <select id="pick"></select></label>
  <span>View:
    <button data-view="traced" class="on">ray-traced</button><button
      data-view="predicted">predicted</button><button data-view="error">error</button>
  </span>
  <span>Vertical exaggeration:
    <button data-vx="1">1x</button><button data-vx="2" class="on">2x</button><button
      data-vx="3">3x</button>
  </span>
  <span id="info"></span>
</header>
<div id="plot"></div>
<footer id="caption">__CAPTION__</footer>
<script>
const DATA = __DATA__;
const N = 128, CELL_KM = 0.04, NO_POWER = -32768;
const axis = Array.from({length: N}, (_, i) => +((i + 0.5) * CELL_KM - N * CELL_KM / 2).toFixed(3));
async function inflate(b64) {
  const bytes = Uint8Array.from(atob(b64), c => c.charCodeAt(0));
  const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream('deflate'));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}
async function int16(b64) {
  const bytes = await inflate(b64), view = new DataView(bytes.buffer), out = new Array(N * N);
  for (let i = 0; i < N * N; i++) {
    const v = view.getInt16(2 * i, true);
    out[i] = v === NO_POWER ? null : v / 10;
  }
  return out;
}
async function bits(b64) {
  const bytes = await inflate(b64), out = new Array(N * N);
  for (let i = 0; i < N * N; i++) out[i] = (bytes[i >> 3] >> (7 - (i & 7))) & 1;
  return out;
}
const grid = flat => Array.from({length: N}, (_, r) => flat.slice(r * N, (r + 1) * N));
let view = 'traced', vx = 2, cur = 0;
const pick = document.getElementById('pick');
DATA.cases.forEach((c, i) => {
  const t = DATA.terrains[c.terrain], o = document.createElement('option');
  o.value = i;
  o.textContent =
    `terrain ${c.terrain} (${t.site_class}), azimuth ${c.azimuth} deg, tilt ${c.tilt} deg`;
  pick.appendChild(o);
});
async function draw() {
  const c = DATA.cases[cur], t = DATA.terrains[c.terrain];
  const [ground, traced, predicted, says] = await Promise.all(
    [int16(t.ground), int16(c.traced), int16(c.predicted), bits(c.says_power)]);
  let values, scale, cmin, cmax, title;
  if (view === 'traced') {
    values = traced; scale = 'Viridis'; cmin = -150; cmax = -60; title = 'path gain (dB)';
  } else if (view === 'predicted') {
    values = predicted.map((v, i) => says[i] ? v : null);
    scale = 'Viridis'; cmin = -150; cmax = -60; title = 'path gain (dB)';
  } else {
    values = predicted.map((v, i) => traced[i] === null ? null : v - traced[i]);
    scale = 'RdBu'; cmin = -__ERR__; cmax = __ERR__; title = 'predicted minus traced (dB)';
  }
  const traces = [
    { type: 'surface', x: axis, y: axis, z: grid(ground), surfacecolor: grid(ground.map(() => 0)),
      colorscale: [[0, '#8a8a8a'], [1, '#8a8a8a']], showscale: false, hoverinfo: 'skip',
      name: 'no signal (grey)', showlegend: true },
    { type: 'surface', x: axis, y: axis,
      z: grid(ground.map((g, i) => values[i] === null ? null : g + 1)),
      surfacecolor: grid(values.map(v => v === null ? 0 : v)),
      colorscale: scale, cmin: cmin, cmax: cmax,
      colorbar: { title: { text: title, side: 'right' }, len: 0.6 },
      hovertemplate: 'east %{x} km<br>north %{y} km<br>%{surfacecolor:.1f} dB<extra></extra>' }
  ];
  const fx = [], fy = [], fz = [], mx = [], my = [], mz = [];
  for (let i = 0; i < N * N; i++) {
    const r = Math.floor(i / N), k = i % N, has = traced[i] !== null;
    if (says[i] && !has) { fx.push(axis[k]); fy.push(axis[r]); fz.push(ground[i] + 15); }
    if (!says[i] && has) { mx.push(axis[k]); my.push(axis[r]); mz.push(ground[i] + 15); }
  }
  traces.push({ type: 'scatter3d', mode: 'markers', x: fx, y: fy, z: fz, hoverinfo: 'skip',
    marker: { size: 2.5, color: '#ff7a00' },
    name: `predicted power, none traced: ${c.false_power} cells` });
  traces.push({ type: 'scatter3d', mode: 'markers', x: mx, y: my, z: mz, hoverinfo: 'skip',
    marker: { size: 2.5, color: '#e040fb' },
    name: `traced power, none predicted: ${c.missed_power} cells` });
  const az = c.azimuth * Math.PI / 180, tl = c.tilt * Math.PI / 180, L = __BEAM__;
  traces.push({ type: 'scatter3d', mode: 'lines', x: [0, 0], y: [0, 0],
    z: [t.site_ground_m, t.antenna_m], line: { color: '#ff3030', width: 8 },
    name: `mast, antenna ${t.antenna_m} m` });
  traces.push({ type: 'scatter3d', mode: 'lines',
    x: [0, L / 1000 * Math.sin(az) * Math.cos(tl)], y: [0, L / 1000 * Math.cos(az) * Math.cos(tl)],
    z: [t.antenna_m, t.antenna_m - L * Math.sin(tl)], line: { color: '#ffd000', width: 5 },
    name: `boresight ${L / 1000} km: azimuth ${c.azimuth} deg, tilt ${c.tilt} deg` });
  document.getElementById('info').textContent =
    `${t.site_class} site on ${t.site_ground_m} m ground (percentile ${t.percentile}); ` +
    `relief ${t.relief_m} m`;
  const lo = Math.min(...ground), hi = Math.max(...ground, t.antenna_m);
  Plotly.react('plot', traces, {
    paper_bgcolor: '#111', font: { color: '#ddd' }, margin: { l: 0, r: 0, t: 0, b: 0 },
    showlegend: true, legend: { x: 0.01, y: 0.98, bgcolor: 'rgba(17,17,17,0.6)' },
    scene: {
      xaxis: { title: { text: 'east (km)' }, color: '#888' },
      yaxis: { title: { text: 'north (km)' }, color: '#888' },
      zaxis: { title: { text: 'height (m)' }, color: '#888', range: [lo - 10, hi + 40] },
      aspectmode: 'manual',
      aspectratio: { x: 1, y: 1, z: vx * (hi - lo + 50) / (N * CELL_KM * 1000) },
      camera: { eye: { x: -0.95, y: -1.1, z: 0.65 } }, bgcolor: '#111'
    },
    uirevision: 'keep'
  }, { responsive: true });
  document.body.dataset.drawn = `${cur}-${view}-${vx}`;
}
pick.onchange = () => { cur = +pick.value; draw(); };
document.querySelectorAll('button[data-view]').forEach(b => b.onclick = () => {
  view = b.dataset.view;
  document.querySelectorAll('button[data-view]').forEach(x => x.classList.toggle('on', x === b));
  draw();
});
document.querySelectorAll('button[data-vx]').forEach(b => b.onclick = () => {
  vx = +b.dataset.vx;
  document.querySelectorAll('button[data-vx]').forEach(x => x.classList.toggle('on', x === b));
  draw();
});
draw();
</script>
</body>
</html>
"""


def viewer_html(data: dict[str, Any], caption: str) -> str:
    """The page, with the data and caption filled in.

    Args:
        data: Output of `viewer_data`.
        caption: Provenance caption shown under the plot (plain text).

    Returns:
        HTML text.
    """
    from sionna_twin_ops.evaluate import ERROR_LIMIT_DB

    escaped = caption.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return (
        PAGE.replace("__PLOTLY__", PLOTLY_URL)
        .replace("__CAPTION__", escaped)
        .replace("__BEAM__", f"{BEAM_LENGTH_M:.0f}")
        .replace("__ERR__", f"{ERROR_LIMIT_DB:.0f}")
        .replace("__DATA__", json.dumps(data, separators=(",", ":")))
    )
