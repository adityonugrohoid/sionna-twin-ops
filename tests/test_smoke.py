"""Environment smoke tests: the CLI runs, and Sionna RT solves a flat tile on llvm."""

import json

import numpy as np
import pytest

from sionna_twin_ops.cli import main

VARIANT = "llvm_ad_mono_polarized"


def test_env_reports_versions(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["env"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert set(report["packages"]) >= {"sionna-rt", "mitsuba", "drjit", "torch"}
    assert VARIANT in report["mitsuba_variants"]


@pytest.mark.sionna
def test_flat_tile_solves_on_llvm() -> None:
    import mitsuba as mi

    # Set before importing sionna.rt, which would otherwise try cuda first.
    mi.set_variant(VARIANT)
    from sionna.rt import (
        ITURadioMaterial,
        PlanarArray,
        RadioMapSolver,
        SceneObject,
        Transmitter,
        load_scene,
    )

    assert mi.variant() == VARIANT

    # A 2 x 2 km flat square at z = 0, two triangles.
    half = 1000.0
    vertices = np.array(
        [[-half, -half, 0.0], [half, -half, 0.0], [half, half, 0.0], [-half, half, 0.0]],
        dtype=np.float32,
    )
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.uint32)
    mesh = mi.Mesh("ground", len(vertices), len(faces))
    params = mi.traverse(mesh)
    params["vertex_positions"] = mi.Float(vertices.ravel())
    params["faces"] = mi.UInt32(faces.ravel())
    params.update()

    scene = load_scene()
    scene.frequency = 1.8e9
    scene.edit(
        add=SceneObject(
            mi_mesh=mesh,
            name="ground",
            radio_material=ITURadioMaterial("ground-mat", "medium_dry_ground", thickness=1.0),
        )
    )
    array = PlanarArray(
        num_rows=1, num_cols=1, vertical_spacing=0.5, pattern="iso", polarization="V"
    )
    scene.tx_array = array
    scene.rx_array = array
    scene.add(Transmitter("tx", position=mi.Point3f(0.0, 0.0, 30.0)))

    radio_map = RadioMapSolver()(
        scene,
        center=mi.Point3f(0.0, 0.0, 1.5),
        orientation=mi.Point3f(0.0, 0.0, 0.0),
        size=mi.Point2f(1000.0, 1000.0),
        cell_size=mi.Point2f(50.0, 50.0),
        samples_per_tx=10**5,
        max_depth=1,
        seed=1,
    )
    gain = np.asarray(radio_map.path_gain.numpy())[0]

    assert gain.shape == (20, 20)
    assert np.isfinite(gain).all()
    assert (gain > 0).mean() > 0.9
    # Gain falls with distance from the transmitter on a flat plane.
    assert gain[10, 10] > gain[0, 0]
