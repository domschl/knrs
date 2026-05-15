from __future__ import annotations

import os
import sys
import json
import subprocess
import argparse
import logging
import warnings
import gc
from typing import Any, Dict, List, Optional

import torch
from pypdf import PdfReader, PdfWriter
from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.accelerator_options import AcceleratorOptions

# Setup logging
from rich.console import Console
from rich.logging import RichHandler

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[
        RichHandler(
            console=Console(stderr=True),
            rich_tracebacks=True,
            show_path=False,
            markup=False,
        )
    ],
)
logger = logging.getLogger("md_converter")

DEFAULT_CONFIG: Dict[str, str] = {
    "device": "auto"
}

def get_platform_config() -> Dict[str, Any]:
    config_file = os.path.expanduser("~/.config/knrs/converter_config_md_converter.json")
    try:
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                return json.load(f)
    except Exception as e:
        logger.error(f"Error loading platform config: {e}")

    try:
        os.makedirs(os.path.dirname(config_file), exist_ok=True)
        with open(config_file, 'w') as f:
            json.dump(DEFAULT_CONFIG, f, indent=4)
    except Exception as e:
        logger.error(f"Failed to create default config at {config_file}: {e}")

    return DEFAULT_CONFIG.copy()

def _get_device(config_device: str = "auto") -> str:
    if config_device and config_device != "auto":
        return config_device

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def convert(source_file: str, destination_file: str) -> None:
    sys.setrecursionlimit(10000)
    
    if not os.path.exists(source_file):
        logger.error(f"Source file does not exist: {source_file}")
        sys.exit(1)
        
    ext = source_file.lower()
    
    if ext.endswith('.epub'):
        try:
            temp_dest = destination_file + ".tmp"
            result = subprocess.run(
                ['pandoc', source_file, '-t', 'gfm', '-o', temp_dest],
                capture_output=True, text=True, check=True
            )
            os.replace(temp_dest, destination_file)
            logger.info(f"Successfully converted EPUB: {source_file} to {destination_file}")
            sys.exit(0)
        except subprocess.CalledProcessError as e:
            logger.error(f"Pandoc conversion failed: {e}")
            sys.exit(1)
        except FileNotFoundError:
            logger.error("Error: pandoc is not installed or not in PATH.")
            sys.exit(1)
            
    elif ext.endswith('.pdf'):
        try:
            # Silence specific noisy loggers that spam non-fatal errors or progress
            logging.getLogger("docling.models.inference_engines.vlm.transformers_engine").setLevel(logging.WARNING)
            logging.getLogger("docling.models.inference_engines.vlm.auto_inline_engine").setLevel(logging.WARNING)
            logging.getLogger("httpx").setLevel(logging.WARNING)
            warnings.filterwarnings("ignore", message="The tied weights mapping")
            
            # Using Docling default OCR as requested (avoiding tesseract which corrupts diacritics).
            # Enable math formula translation to latex embedded in markdown
            pipeline_options = PdfPipelineOptions()
            pipeline_options.do_formula_enrichment = True
            
            config = get_platform_config()
            device = _get_device(config.get("device", "auto"))
            pipeline_options.accelerator_options = AcceleratorOptions(device=device)
            
            reader = PdfReader(source_file)
            total_pages = len(reader.pages)
            chunk_size = 200
            
            markdown_chunks: List[str] = []
            target_dir = os.path.dirname(destination_file)
            if target_dir and not os.path.exists(target_dir):
                os.makedirs(target_dir, exist_ok=True)
                
            for start_page in range(0, total_pages, chunk_size):
                end_page = min(start_page + chunk_size, total_pages)
                logger.info(f"Processing chunk: pages {start_page + 1} to {end_page} of {total_pages}...")
                
                # Write chunk to temp file in the same directory as source
                temp_pdf_path = f"{source_file}.chunk_{start_page}_{end_page}.pdf"
                writer = PdfWriter()
                for i in range(start_page, end_page):
                    writer.add_page(reader.pages[i])
                    
                with open(temp_pdf_path, "wb") as f_out:
                    writer.write(f_out)
                
                # Process chunk
                converter_instance = DocumentConverter(
                    format_options={
                        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
                    }
                )
                try:
                    result = converter_instance.convert(temp_pdf_path)
                    markdown_content: str = result.document.export_to_markdown()
                    markdown_chunks.append(markdown_content)
                    
                    if hasattr(result, 'input') and hasattr(result.input, '_backend'):
                        result.input._backend.unload()
                        
                finally:
                    # Clean up temp file and force garbage collection
                    if os.path.exists(temp_pdf_path):
                        os.remove(temp_pdf_path)
                    del converter_instance
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    elif hasattr(torch, "xpu") and torch.xpu.is_available():
                        torch.xpu.empty_cache()

            # Write combined result atomically
            temp_dest = destination_file + ".tmp"
            with open(temp_dest, 'w', encoding='utf-8') as f:
                f.write("\n\n".join(markdown_chunks))
            os.replace(temp_dest, destination_file)
                
            logger.info(f"Successfully converted document: {source_file} to {destination_file}")
            sys.exit(0)
            
        except Exception as e:
            logger.error(f"Docling conversion failed: {e}")
            sys.exit(1)
    else:
        logger.error(f"Error: Unsupported file extension for {source_file}. Only .pdf and .epub are supported.")
        sys.exit(1)

def main() -> None:
    parser = argparse.ArgumentParser(description="Convert PDF/EPUB to Markdown (Unified)")
    parser.add_argument("source", nargs="?", help="Path to the source file (e.g. .pdf, .epub)")
    parser.add_argument("destination", nargs="?", help="Path to the destination Markdown file")
    parser.add_argument("--capabilities", action="store_true", help="Print backend capabilities as JSON")
    
    args = parser.parse_args()

    if args.capabilities:
        cap: Dict[str, Any] = {
            "name": "md_converter",
            "type": "converter",
            "config_file": "converter_config_md_converter.json",
            "platform": "any",
            "validated_models": ["pandoc", "docling"],
            "available_models": ["pandoc", "docling"],
            "parameters": {
                "device": {"type": "str"}
            },
        }
        print(json.dumps(cap))
        sys.exit(0)

    if not args.source or not args.destination:
        parser.error("source and destination are required unless --capabilities is passed")

    config = get_platform_config()
    device = _get_device(config.get("device", "auto"))
    logger.info("Using device: %s", device)
    convert(args.source, args.destination)

if __name__ == "__main__":
    main()
