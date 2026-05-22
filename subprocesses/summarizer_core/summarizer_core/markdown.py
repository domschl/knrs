from __future__ import annotations

from typing import Any

import yaml

def split_header_content(text: str) -> tuple[str, str]:
    if not text.startswith("---\n"):
        return ("", text)
    parts = text.split("---\n", 2)
    if len(parts) < 3:
        return ("", text)
    return (parts[1], parts[2])

def parse_markdown(md_text: str) -> tuple[dict[str, Any], str]:
    frontmatter, content = split_header_content(md_text)
    try:
        yaml_metadata: dict[str, Any] = yaml.safe_load(frontmatter) if frontmatter else {}
    except Exception:
        yaml_metadata = {}
    return yaml_metadata, content

def assemble_markdown(metadata: dict[str, Any] | None, md_text: str) -> str:
    if metadata is None:
        return md_text
        
    filtered_metadata: dict[str, Any] = {}
    for k, v in metadata.items():
        if isinstance(v, list) and len(v) == 0:
            continue
        if isinstance(v, str) and v == "":
            continue
        filtered_metadata[k] = v
        
    header: str = yaml.dump(filtered_metadata, default_flow_style=False, indent=2)
    if not header.endswith("\n"):
        header += "\n"
    return f"---\n{header}---\n{md_text}"

import re

def get_answer_from_output(text: str) -> str:
    if "<channel|>" in text:
        text = text.split("<channel|>")[-1]
        
    # Strip <think>...</think> tags if present
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    
    # Handle unclosed <think> tag if it was cut off
    if "<think>" in text:
        text = text.split("<think>")[-1]
        if "</think>" in text:
            text = text.split("</think>")[-1]
        else:
            text = ""
            
    return text.strip()
