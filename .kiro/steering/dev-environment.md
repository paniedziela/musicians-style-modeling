---
inclusion: always
---

# Development Environment

The Python environment for this project is already configured. Do NOT recreate it, do NOT reinstall dependencies, and do NOT set `PYTHONPATH`.

## Interpreter

- Python 3.10.20 (conda-based venv; `home = .venv`).
- The shell is **PowerShell**. Use `;` as the command separator (not `&` or `&&`).

**Preferred: activate the venv first, then use `python` / `pytest` directly.**

```
.venv\Scripts\Activate.ps1
python --version
```

Activation persists for the rest of an interactive terminal session, so commands stay clean and readable (`python -m pytest`, `pip ...`). You (the user) do not need to pre-activate; the agent activates it at the start of a session when needed.

**Fallback: when the venv is not active** (a fresh terminal, or a background/`start`ed process, begins unactivated), call the interpreter by its bare absolute path:

```
.venv\Scripts\python.exe -m pytest
```

No quotes and no `&` are needed: this path contains no spaces. PowerShell's call operator `&` plus quoting is only required when a command path contains spaces (e.g. `& 'C:\Program Files\...\python.exe'`). It is not a security or trust mechanism, just a quoting artifact, so prefer the bare path for readability.

## Package install

The package is installed **editable** (`pip install -e . --no-deps`), so `import musicians_style` resolves from `src/musicians_style` regardless of current directory or environment variables. There is an `__editable__.musicians_style-0.1.0.pth` in site-packages. Do not rely on `PYTHONPATH=src`.

## Dependencies (already installed, pinned)

torch 2.2.2, numpy 1.26.4, scipy 1.13.1, mido 1.3.2, pretty_midi 0.2.10, structlog 24.1.0, PyYAML 6.0.1, matplotlib 3.8.4, pyFluidSynth 1.3.3, yt-dlp 2024.8.6, pytest 8.2.2, hypothesis 6.103.2.

All dependencies from `requirements.txt` / `pyproject.toml` are present. If a genuinely new dependency is needed, install it into this venv with the absolute interpreter path and pin its version.

## Running tests

From the project root `.`, with the venv active:

```
python -m pytest -q
```

Or, if the venv is not active, with the bare absolute path:

```
.venv\Scripts\python.exe -m pytest -q
```

`pytest.ini` is the single source of truth for pytest config (`testpaths = tests`, markers `property` and `slow`). As of the last full run, all existing tests pass.
