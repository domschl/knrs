# knrs

> [!WARNING]
> This project is under active development. Everything is subject to change at any time.

LLM-enabled knowledge-base wiki with Calibre integration, VectorDB, and local-first Wiki and research agents

## Installation

To sync dependencies and initialize the subprocess backends, run:

```bash
python3 install.py
```

*Note: In network-isolated or offline environments, you can run:*
```bash
python3 install.py --offline
```

## How to Start

To start the interactive REPL:

```bash
uv run knrs
```

To view the resolved configuration:

```bash
uv run knrs config
```

To list all available CLI commands:

```bash
uv run knrs --help
```
