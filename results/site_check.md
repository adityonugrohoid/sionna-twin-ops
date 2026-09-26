# Site placement check (spec S2)

Synthetic terrain ids 0 to 59, 40 m grid, each id placed with its own class. Percentile: midrank percentile of the site height among the vertices of its 5.12 km map. Written by `twin site-check`.

| class | ids | placed | skipped | p10 | median | p90 | skipped ids |
|---|---|---|---|---|---|---|---|
| hilltop | 20 | 19 | 1 | 92.0 | 96.8 | 100.0 | 42 |
| slope | 20 | 20 | 0 | 36.7 | 48.3 | 64.2 | none |
| valley | 20 | 16 | 4 | 0.0 | 0.5 | 10.0 | 2, 26, 35, 56 |
