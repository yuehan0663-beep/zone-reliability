# Measured resources and wall times

Hardware: AMD Ryzen 7 8745HS, 8 cores / 16 logical processors; 33,603,244,032
bytes installed RAM (31.3 GiB); CPU execution; no GPU required. Reference Python
3.11.9 and exact installed dependencies are in requirements.txt/environment.yml.

| Operation | Measured seconds | Scope |
|---|---:|---|
| Synthetic quickstart --fast | 191.113583 | Five model families, all phases including fixed-backbone attribution |
| Synthetic full quickstart | 176.610907 | All 16 model families, all phases including attribution |
| Added real-data attribution anchor verification | 67.785905 | Eight scalar comparisons / six new gate groups |
| Previously completed full reference-paper workflow | Approximately 110 | Historical estimate from start/cache file timestamps; not a new instrumented run |

Fast/full times are individual observed wall times, not a benchmark comparing
modes: model selection, process load and fixed bootstrap/attribution work affect
elapsed time. Full reference fitting was not repeated for v2. Public-table and
map checks consume released derived CSVs and do not rerun private-data models.
Final repository size (excluding .git and generated caches) is recorded in the
external code_availability.md and final acceptance report.
