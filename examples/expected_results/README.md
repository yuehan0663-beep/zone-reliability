# Measured synthetic reference outputs

These two folders contain completed A-environment fast/full tutorial outputs,
generated entirely from distributions with seed 20261009. They contain no real
log or core rows. `runtime.json` records the measured wall time and model count.
Floating-point model outputs may vary across platforms; these files are examples,
not exact-byte CI targets. Use `verify_synthetic_outputs.py` to check structural,
group separation and finite-output requirements on a freshly generated run.
