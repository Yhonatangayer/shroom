# Benchmarks

Validation experiments that reproduce the encoder-convergence figures from the SHroom
paper. Each script sweeps the spherical-harmonics (SH) order and reports how the encoder
error decays, validating the accuracy of the encoders against high-order references.

| Script | What it validates |
|--------|-------------------|
| `asm_convergence.py` | Ambisonics Signal Matching (ASM) encoder error vs. SH order. |
| `bsm_convergence.py` | Binaural Signal Matching (BSM) encoder error vs. SH order (against a MATLAB reference). |
| `aa_magls_convergence.py` | Array-aware MagLS binaural magnitude error vs. SH order. |

## Requirements

These scripts depend on the optional **`shroom_dev`** companion package (evaluation
metrics in `shroom_dev.errors` and plotting in `shroom_dev.plot`). Install the `[dev]`
extra from a checkout of the repository:

```bash
pip install -e ".[dev]"
```

`shroom_dev` is **not** required to use the core `shroom` library — only to run these
benchmarks and the examples.

## Running

```bash
python benchmarks/asm_convergence.py
python benchmarks/bsm_convergence.py
python benchmarks/aa_magls_convergence.py
```

Each script writes its figures to `benchmarks/figures/`.
