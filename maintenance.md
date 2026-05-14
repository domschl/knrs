## Submodules

Inital installation of submodules:
```bash
./install.py [--xpu] [--help]
```

Note: `--xpu` installs Intel XPU optimised torch versions of torch, for other hardware platforms, recognition of hw is automatic.

### First time, only required to reference old code repositories

Note: this is _only required for reference, the functionality of the current project is self-contained, and does not depend on the submodules.

```bash
git submodule --init
git submodule --update
```

Fetch changes from upstream projects:

```bash
git submodule foreach git pull
```

### Llama.cpp

Note: `--embeddings` is required for the embedder.

```
ExecStart=llama-server --models-dir ${XDG_DATA_HOME}/GGUF --jinja --port 8180 --host 0.0.0.0 --fit on --embeddings --models-max 1 --sleep-idle-seconds 180
```

### Intel XPU (torch)

The subprocesses `embedder_hf` and `md_converter` use `torch` from PyPI by default.
On Intel XPU hardware (e.g. Arc/Ponte Vecchio GPUs), install the XPU-optimised
build manually after syncing:

```bash
# Step 1: create/sync the normal environment
cd knrs/subprocesses/embedder_hf
uv sync

# Step 2: overlay the XPU torch on top
uv pip install torch --upgrade --index-url https://download.pytorch.org/whl/xpu
```

Repeat for `md_converter`:

```bash
cd knrs/subprocesses/md_converter
uv sync
uv pip install torch --upgrade --index-url https://download.pytorch.org/whl/xpu
```

> [!NOTE]
> The XPU index is intentionally **not** declared in the `pyproject.toml` files, because
> its wheels are Linux-only and would break `uv sync` on macOS. XPU users must
> run the `uv pip install` overlay step manually.
