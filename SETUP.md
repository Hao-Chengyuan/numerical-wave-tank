# Setup Guide

Step-by-step instructions for getting this project running on a new machine.

## Prerequisites

You need [uv](https://docs.astral.sh/uv/) — a fast Python package manager. Install it once per machine:

**macOS / Linux**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows (PowerShell)**
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.sh | iex"
```

After installing, restart your terminal so `uv` is on your PATH.

---

## Clone and set up

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/numerical-wave-tank.git
cd numerical-wave-tank

# 2. Create the virtual environment and install all dependencies
#    (uv reads pyproject.toml and uv.lock to reproduce the exact environment)
uv sync

# 3. Verify everything works
uv run python fun_test.py
```

That's it — the `.venv/` folder is created automatically inside the project directory.

---

## Daily use

Always prefix commands with `uv run` so they use the project's environment, or activate the venv first:

```bash
# Option A — prefix each command (no activation needed)
uv run python wave_gen_new.py
uv run python fun_test.py
uv run jupyter notebook

# Option B — activate once per terminal session, then run normally
source .venv/bin/activate          # macOS / Linux
.venv\Scripts\activate             # Windows

python fun_test.py
jupyter notebook
```

---

## Adding new packages

```bash
uv add <package-name>        # adds to pyproject.toml and updates uv.lock
```

Commit both `pyproject.toml` and `uv.lock` after adding packages so all devices stay in sync.

---

## Keeping in sync across devices

```bash
git pull                     # fetch latest code
uv sync                      # update the environment to match uv.lock
```

---

## IDE integration

**VS Code** — install the [Python extension](https://marketplace.visualstudio.com/items?itemName=ms-python.python), then select the interpreter at `.venv/bin/python` (Cmd+Shift+P → "Python: Select Interpreter").

**JetBrains (PyCharm / DataSpell)** — go to Settings → Project → Python Interpreter → Add → Existing environment → point to `.venv/bin/python`.
