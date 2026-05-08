## Submodules

This project imports older projects with overlapping functionality as submodules for references.

### First time

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

### torch

XPU:

```bash
 uv pip install torch --upgrade --index-url https://download.pytorch.org/whl/xpu
```
