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

Because the repo is **private**, GitHub needs to verify your identity before it lets you clone. The recommended way is a Personal Access Token (PAT).

### 1. Create a Personal Access Token (once per GitHub account)

1. Go to **github.com → Settings → Developer settings → Personal access tokens → Tokens (classic)**
2. Click **Generate new token (classic)**
3. Give it a name (e.g. `dev-laptop`), set an expiry, and tick the **`repo`** scope
4. Click **Generate token** and copy it — you won't see it again

### 2. Clone, set up, and verify

```bash
# Clone — paste your token when Git asks for a password
#         (use your GitHub username when asked for a username)
git clone https://github.com/Hao-Chengyuan/numerical-wave-tank.git
cd numerical-wave-tank

# Create the virtual environment and install all dependencies
uv sync

# Verify everything works
uv run python src/fun_test.py
```

> **Tip — save credentials so you aren't asked every time:**
> ```bash
> git config --global credential.helper store
> ```
> After entering your token once, Git saves it to `~/.git-credentials` and reuses it automatically.

That's it — the `.venv/` folder is created automatically inside the project directory.

---

## Daily use

Always prefix commands with `uv run` so they use the project's environment, or activate the venv first:

```bash
# Option A — prefix each command (no activation needed)
uv run python src/wave_gen_new.py
uv run python src/fun_test.py
uv run jupyter notebook

# Option B — activate once per terminal session, then run normally
source .venv/bin/activate          # macOS / Linux
.venv\Scripts\activate             # Windows

python src/fun_test.py
jupyter notebook
```

---

## Adding new packages

```bash
uv add <package-name>        # adds to pyproject.toml and updates uv.lock
```

Commit both `pyproject.toml` and `uv.lock` after adding packages so all devices stay in sync.

---

## Editing files and pushing back to GitHub

The standard edit → commit → push loop:

```bash
# 1. Always pull first to get the latest changes before you start editing
git pull

# 2. Edit files as needed, then stage the ones you changed
git add src/wave_gen_new.py          # stage a specific file
git add src/                         # or stage everything in src/

# 3. Commit with a short message describing what you changed
git commit -m "describe what you changed"

# 4. Push to GitHub
git push
```

If you edited on two devices without syncing, Git may report a conflict on `git pull`. Resolve it by opening the flagged file, choosing which version to keep, then:

```bash
git add <conflicted-file>
git commit -m "resolve merge conflict"
git push
```

---

## Keeping in sync across devices

```bash
git pull                     # fetch latest code from GitHub
uv sync                      # update the environment if pyproject.toml changed
```

---

## IDE integration

**VS Code** — install the [Python extension](https://marketplace.visualstudio.com/items?itemName=ms-python.python), then select the interpreter at `.venv/bin/python` (Cmd+Shift+P → "Python: Select Interpreter").

**JetBrains (PyCharm / DataSpell)** — go to Settings → Project → Python Interpreter → Add → Existing environment → point to `.venv/bin/python`.
