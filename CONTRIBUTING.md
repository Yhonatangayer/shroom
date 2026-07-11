# Contributing to shroom

Contributions are welcome — bug reports, feature requests, documentation fixes, and pull
requests.

## Reporting issues and seeking support

- **Bugs and feature requests:** open an issue at
  <https://github.com/Yhonatangayer/shroom/issues>. Please include your OS, Python version,
  `shroom`/`pyshroom` version, and a minimal script that reproduces the problem.
- **Questions and support:** open an issue with the `question` label, or start a discussion
  on the repository. There are no private support channels — please keep questions public so
  others benefit from the answers.

## Development setup

Clone the repository and install it in editable mode with the development extras:

```bash
git clone https://github.com/Yhonatangayer/shroom.git
cd shroom
pip install -e ".[dev]"
```

This installs the core library plus the `shroom_dev` companion package and the tooling
used for tests, examples, and benchmarks (`pytest`, `black`, `sounddevice`, `pyyaml`).

## Running the tests

```bash
pytest
```

All tests should pass before you open a pull request. If you add functionality, please add
tests that cover it under `tests/`.

## Coding style

- Follow PEP 8. The project uses [`black`](https://black.readthedocs.io/) (installed with the
  `[dev]` extra); run `black src tests examples benchmarks` before committing.
- Prefer type hints and docstrings on public functions and classes — the docstrings serve as
  the API reference.
- Keep new code consistent with the surrounding modules' naming and structure.

## Pull requests

1. Fork the repository and create a feature branch.
2. Make your change with accompanying tests and updated docstrings.
3. Ensure `pytest` passes and the code is `black`-formatted.
4. Open a pull request describing the change and the motivation.

By contributing, you agree that your contributions will be licensed under the MIT License
of this project.
