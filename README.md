# sionna-twin-ops

A learned surrogate of a ray tracer for one radio sector on hilly
terrain. Sionna RT computes coverage maps over procedurally generated
hills, ridges and valleys; a small neural network learns to predict them
on terrain it has never seen, and a search uses it to choose tilt,
azimuth and power. Every result is measured
against the ray tracer and two classic propagation baselines, and the
terrain is synthetic, not a real place.

## Quickstart

```bash
git clone https://github.com/adityonugrohoid/sionna-twin-ops.git
cd sionna-twin-ops
```

The repo is being built. The terrain generator and the ray-tracing
pipeline land in the next pull requests, and this section will then
carry the commands that run them.

Built on NVIDIA Sionna RT; not affiliated with NVIDIA.

## License

MIT, see [LICENSE](LICENSE).

## Author

Adityo Nugroho ([adityonugroho.com](https://adityonugroho.com)),
building with a Claude Code agentic workflow.
