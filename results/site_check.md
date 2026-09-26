# Site placement check (spec S2)

Synthetic terrain ids 0 to 59, 40 m grid, each id placed with its own class. Percentile: midrank percentile of the site height among the vertices of its 5.12 km map. Written by `twin site-check`.

| class | ids | placed | skipped | p10 | median | p90 | skipped ids |
|---|---|---|---|---|---|---|---|
| hilltop | 20 | 14 | 6 | 87.4 | 93.2 | 97.5 | 0, 12, 15, 39, 42, 48 |
| slope | 20 | 20 | 0 | 35.8 | 55.2 | 64.4 | none |
| valley | 20 | 14 | 6 | 0.1 | 5.8 | 12.4 | 2, 26, 35, 41, 56, 59 |
