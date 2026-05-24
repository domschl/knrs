from __future__ import annotations

import json
import logging
import platform
import re
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from agent.engine import AgentSession
from benchmark.sample_docs import (
    generate_minimal_epub,
    generate_minimal_md,
    generate_minimal_pdf,
)
from benchmark.system_info import get_system_info
from config import KnrsConfig
from repl.backends import BackendManager
from vector.engine import EmbedderSession

logger = logging.getLogger(__name__)


# ─── Verification & Soft-Failure Check Helpers ─────────────────────────────────

def check_refusal_or_excuse(text: str) -> bool:
    """Return True if the text matches common refusal or excuse patterns."""
    refusal_patterns = [
        r"\bcannot summarize\b",
        r"\bunable to summarize\b",
        r"\bnot able to summarize\b",
        r"\bsorry, but\b",
        r"\bas an AI\b",
        r"\bcould summarize, because\b",
        r"\bcannot fulfill\b",
        r"\bno context provided\b",
        r"\bcontext does not contain\b",
        r"\bno information provided\b",
        r"\bnot mention\b",
        r"\bdoes not provide enough information\b",
        r"\bnot enough context\b",
    ]
    text_lower = text.lower()
    for pattern in refusal_patterns:
        if re.search(pattern, text_lower):
            return True
    return False


def calculate_content_overlap(source_text: str, summary_text: str) -> set[str]:
    """Calculate the overlap of content words between source and summary."""
    stop_words = {
        "the", "a", "an", "and", "or", "but", "if", "then", "else", "when", 
        "at", "by", "for", "with", "about", "against", "between", "into", 
        "through", "during", "before", "after", "above", "below", "to", 
        "from", "up", "down", "in", "out", "on", "off", "over", "under", 
        "again", "further", "then", "once", "here", "there", "all", "any", 
        "both", "each", "few", "more", "most", "other", "some", "such", 
        "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very",
        "is", "was", "were", "be", "been", "being", "have", "has", "had", "having",
        "do", "does", "did", "doing", "can", "could", "will", "would", "should",
        "this", "that", "these", "those"
    }
    
    def get_content_words(t: str) -> set[str]:
        words = re.findall(r"\b[a-z]{3,}\b", t.lower())
        return {w for w in words if w not in stop_words}
        
    src_words = get_content_words(source_text)
    sum_words = get_content_words(summary_text)
    return sum_words.intersection(src_words)


def merge_run_record(history: list[dict[str, Any]], new_record: dict[str, Any]) -> list[dict[str, Any]]:
    """Merge a new run record into history, updating or creating the entry for this host.

    - If a backend fails in the new run, numeric metrics are zeroed.
    - If a backend passes in both the old and new runs, we apply a moving average
      (0.8 * old + 0.2 * new) to numeric metrics, provided the old run didn't fail.
    - Results for backends not tested in the new run are left unaltered.
    """
    current_hostname = new_record.get("hostname")

    # 1. Preprocess new record's results to zero out failed metrics
    preprocessed_results = []
    for res in new_record.get("results", []):
        res_copy = res.copy()
        if res_copy.get("pass_fail") == "fail":
            res_copy["load_time_sec"] = 0.0
            res_copy["latency_sec"] = 0.0
            res_copy["throughput"] = 0.0
        preprocessed_results.append(res_copy)

    # 2. Find existing record for this host
    existing_record = None
    existing_idx = -1
    for i, entry in enumerate(history):
        if isinstance(entry, dict) and entry.get("hostname") == current_hostname:
            existing_record = entry
            existing_idx = i
            break

    if existing_record is None:
        # No existing record for this host, append the preprocessed new record
        new_record_copy = new_record.copy()
        new_record_copy["results"] = preprocessed_results
        history.append(new_record_copy)
        return history

    # 3. Merge results
    existing_results = existing_record.get("results", [])
    existing_map = {}
    for res in existing_results:
        key = (res.get("backend"), res.get("backend_type"), res.get("task_name"))
        existing_map[key] = res.copy()

    for new_res in preprocessed_results:
        key = (new_res.get("backend"), new_res.get("backend_type"), new_res.get("task_name"))
        old_res = existing_map.get(key)

        if new_res.get("pass_fail") == "pass":
            if (
                old_res is not None
                and old_res.get("pass_fail") == "pass"
                and old_res.get("latency_sec", 0.0) > 0.0
            ):
                # Calculate moving average (0.8 * old + 0.2 * new)
                new_res["load_time_sec"] = round(
                    0.8 * old_res.get("load_time_sec", 0.0) + 0.2 * new_res.get("load_time_sec", 0.0),
                    6
                )
                new_res["latency_sec"] = round(
                    0.8 * old_res.get("latency_sec", 0.0) + 0.2 * new_res.get("latency_sec", 0.0),
                    6
                )
                new_res["throughput"] = round(
                    0.8 * old_res.get("throughput", 0.0) + 0.2 * new_res.get("throughput", 0.0),
                    6
                )
            existing_map[key] = new_res
        else:
            # failure, overwrite directly
            existing_map[key] = new_res

    # 4. Create updated record
    updated_record = {
        "timestamp": new_record.get("timestamp"),
        "hostname": current_hostname,
        "os": new_record.get("os"),
        "cpu": new_record.get("cpu"),
        "accelerator": new_record.get("accelerator"),
        "results": list(existing_map.values()),
    }

    history[existing_idx] = updated_record
    return history


# ─── Benchmark Runner ─────────────────────────────────────────────────────────

class BenchmarkRunner:
    """Orchestrates, runs, and logs benchmarks for knrs subprocesses."""

    def __init__(self, workspace_root: Path, results_path: Path | None = None) -> None:
        self.workspace_root = workspace_root
        if results_path is not None:
            self.results_path = results_path
        else:
            from config import load_config
            try:
                cfg = load_config()
                benchmark_dir = cfg.benchmark_path
            except Exception:
                benchmark_dir = Path("~/.config/knrs/benchmarks").expanduser().resolve()
            
            benchmark_dir.mkdir(parents=True, exist_ok=True)
            self.results_path = benchmark_dir / f"benchmark_{socket.gethostname()}.json"
        self.backend_manager = BackendManager(workspace_root / "subprocesses")
        self.sys_info = get_system_info()
        
        # Default test content
        self.test_title = "Rome: Foundation and Expansion"
        self.test_text = (
            "Rome was founded in 753 BC by Romulus. According to legend, Romulus and "
            "his twin brother Remus were raised by a she-wolf. The city grew from a small "
            "agricultural settlement on the banks of the Tiber River into a massive empire. "
            "During the Punic Wars, Rome defeated Carthage and gained control of the Mediterranean. "
            "Julius Caesar conquered Gaul and was later assassinated in 44 BC. Octavian became "
            "Augustus, the first Emperor, inaugurating the Pax Romana period of relative peace."
        )

    def run_all(self, filter_type: str | None = None, filter_backend: str | None = None) -> dict[str, Any]:
        """Run discovered benchmarks, filter by type/backend, and save results."""
        logger.info("Starting knrs benchmark run...")
        logger.info("System details: CPU=%s, Accel=%s", self.sys_info["cpu"], self.sys_info["accelerator"])

        from config import load_config
        active_summarizer: str | None = None
        active_embedder: str | None = None
        active_agent: str | None = None
        try:
            user_cfg = load_config()
            active_summarizer = user_cfg.summarizer_name
            active_embedder = user_cfg.embedder_name
            active_agent = user_cfg.agent_backend_name
        except Exception as e:
            logger.warning("Could not load user config to detect active backends: %s. Using defaults.", e)
            active_summarizer = "summarizer_linux"
            active_embedder = "embedder_hf"
            active_agent = "agent_api"

        results = []

        with TemporaryDirectory() as tmpdir_str:
            tmpdir = Path(tmpdir_str)
            cfg = self._make_temp_config(tmpdir)

            # 1. Benchmark Converters
            if not filter_type or filter_type == "converter":
                if not filter_backend or filter_backend == "md_converter":
                    results.extend(self._benchmark_converters(cfg, tmpdir))

            # 2. Benchmark Summarizers
            if not filter_type or filter_type == "summarizer":
                summarizers = self.backend_manager.get_backends("summarizer")
                for name in summarizers:
                    should_run = False
                    if filter_backend:
                        should_run = (filter_backend == name)
                    else:
                        should_run = (name == active_summarizer)
                    
                    if should_run:
                        results.extend(self._benchmark_summarizer(name, cfg, tmpdir))

            # 3. Benchmark Embedders
            if not filter_type or filter_type == "embedder":
                embedders = self.backend_manager.get_backends("embedder")
                for name in embedders:
                    should_run = False
                    if filter_backend:
                        should_run = (filter_backend == name)
                    else:
                        should_run = (name == active_embedder)
                    
                    if should_run:
                        results.extend(self._benchmark_embedder(name, cfg))

            # 4. Benchmark Agents
            if not filter_type or filter_type == "agent":
                agents = self.backend_manager.get_backends("agent")
                for name in agents:
                    should_run = False
                    if filter_backend:
                        should_run = (filter_backend == name)
                    else:
                        should_run = (name == active_agent)
                    
                    if should_run:
                        results.extend(self._benchmark_agent(name, cfg))

        # Compile final report object
        run_record = {
            "timestamp": datetime.now().isoformat(),
            "hostname": self.sys_info["hostname"],
            "os": self.sys_info["os"],
            "cpu": self.sys_info["cpu"],
            "accelerator": self.sys_info["accelerator"],
            "results": results,
        }

        self._save_results(run_record)
        return run_record

    def _make_temp_config(self, tmpdir: Path) -> KnrsConfig:
        """Create a localized KnrsConfig to prevent overwriting user directories."""
        return KnrsConfig(
            calibre_path=tmpdir / "calibre",
            notes_path=tmpdir / "notes",
            knrs_data=tmpdir / "knrs_data",
            wiki_path=tmpdir / "wiki",
            vector_db_path=tmpdir / "vector_db.json",
            benchmark_path=tmpdir / "benchmarks",
        )

    # ─── Converter Benchmarks ──────────────────────────────────────────────────

    def _benchmark_converters(self, cfg: KnrsConfig, tmpdir: Path) -> list[dict[str, Any]]:
        results = []
        script = self.workspace_root / "subprocesses" / "md_converter" / "md_converter.py"
        venv_python = script.parent / ".venv" / "bin" / "python"
        python_exe = str(venv_python) if venv_python.exists() else sys.executable

        # A. PDF Conversion
        pdf_path = tmpdir / "test.pdf"
        md_out_path = tmpdir / "pdf_converted.md"
        generate_minimal_pdf(pdf_path, self.test_text)

        logger.info("Benchmarking md_converter [PDF -> MD]...")
        start_time = time.time()
        try:
            p = subprocess.run(
                [python_exe, str(script), str(pdf_path), str(md_out_path)],
                capture_output=True,
                text=True,
                check=True,
                timeout=120,
            )
            elapsed = time.time() - start_time
            
            # Validation
            passed = md_out_path.exists() and len(md_out_path.read_text(encoding="utf-8").strip()) > 0
            err_msg = None if passed else "Converted Markdown file is empty or missing."
            char_count = len(md_out_path.read_text(encoding="utf-8")) if passed else 0
            throughput = char_count / elapsed if elapsed > 0 else 0
            
            results.append({
                "backend": "md_converter",
                "backend_type": "converter",
                "task_name": "convert_pdf",
                "pass_fail": "pass" if passed else "fail",
                "load_time_sec": 0.0,  # run-to-completion
                "latency_sec": elapsed,
                "throughput": throughput,
                "throughput_units": "chars/sec",
                "error": err_msg,
            })
        except Exception as e:
            results.append({
                "backend": "md_converter",
                "backend_type": "converter",
                "task_name": "convert_pdf",
                "pass_fail": "fail",
                "load_time_sec": 0.0,
                "latency_sec": time.time() - start_time,
                "throughput": 0.0,
                "throughput_units": "chars/sec",
                "error": str(e),
            })

        # B. EPUB Conversion
        epub_path = tmpdir / "test.epub"
        md_out_path = tmpdir / "epub_converted.md"
        generate_minimal_epub(epub_path, self.test_title, self.test_text)

        logger.info("Benchmarking md_converter [EPUB -> MD]...")
        start_time = time.time()
        try:
            p = subprocess.run(
                [python_exe, str(script), str(epub_path), str(md_out_path)],
                capture_output=True,
                text=True,
                check=True,
                timeout=60,
            )
            elapsed = time.time() - start_time
            
            # Validation
            passed = md_out_path.exists() and len(md_out_path.read_text(encoding="utf-8").strip()) > 0
            err_msg = None if passed else "Converted Markdown file is empty or missing."
            char_count = len(md_out_path.read_text(encoding="utf-8")) if passed else 0
            throughput = char_count / elapsed if elapsed > 0 else 0
            
            results.append({
                "backend": "md_converter",
                "backend_type": "converter",
                "task_name": "convert_epub",
                "pass_fail": "pass" if passed else "fail",
                "load_time_sec": 0.0,
                "latency_sec": elapsed,
                "throughput": throughput,
                "throughput_units": "chars/sec",
                "error": err_msg,
            })
        except Exception as e:
            results.append({
                "backend": "md_converter",
                "backend_type": "converter",
                "task_name": "convert_epub",
                "pass_fail": "fail",
                "load_time_sec": 0.0,
                "latency_sec": time.time() - start_time,
                "throughput": 0.0,
                "throughput_units": "chars/sec",
                "error": str(e),
            })

        return results

    # ─── Summarizer Benchmarks ─────────────────────────────────────────────────

    def _benchmark_summarizer(self, name: str, cfg: KnrsConfig, tmpdir: Path) -> list[dict[str, Any]]:
        results = []
        script = self.workspace_root / "subprocesses" / name / f"{name}.py"
        venv_python = script.parent / ".venv" / "bin" / "python"
        python_exe = str(venv_python) if venv_python.exists() else sys.executable

        md_src = tmpdir / f"src_{name}.md"
        sum_dest = tmpdir / f"sum_{name}.md"
        generate_minimal_md(md_src, self.test_title, self.test_text)

        logger.info("Benchmarking summarizer '%s'...", name)
        start_time = time.time()
        try:
            # We enforce summary_max_tokens = 150
            p = subprocess.run(
                [python_exe, str(script), str(md_src), str(sum_dest), "--summary_max_tokens", "150"],
                capture_output=True,
                text=True,
                check=True,
                timeout=300,
            )
            elapsed = time.time() - start_time
            
            # Validation
            passed = True
            err_msg = None
            
            if not sum_dest.exists():
                passed = False
                err_msg = "Summary file was not created."
            else:
                summary_content = sum_dest.read_text(encoding="utf-8").strip()
                if not summary_content:
                    passed = False
                    err_msg = "Summary file is completely empty."
                
                # Check for refusal / soft failure
                elif check_refusal_or_excuse(summary_content):
                    passed = False
                    err_msg = "Soft failure: Summary contains refusal/excuse template phrases."
                
                # Check formatting: frontmatter must exist and contain source_md_hash
                elif not (summary_content.startswith("---") and "source_md_hash" in summary_content):
                    passed = False
                    err_msg = "Soft failure: Summary markdown does not contain valid YAML frontmatter."
                
                # Check keyword overlap (should share at least 3 unique content words with source)
                else:
                    body_content = summary_content
                    if summary_content.startswith("---"):
                        parts = summary_content.split("---", 2)
                        if len(parts) >= 3:
                            body_content = parts[2]
                    body_content = body_content.strip()

                    if not body_content:
                        passed = False
                        err_msg = "Soft failure: Summary body content is empty (possibly cut off in thinking phase)."
                    elif "[summary blocked" in body_content.lower() or "blocked or empty" in body_content.lower():
                        passed = False
                        err_msg = f"Soft failure: Summary generation was blocked or returned empty. Got: '{body_content}'"
                    else:
                        overlap_words = calculate_content_overlap(self.test_text, body_content)
                        if len(overlap_words) < 3:
                            passed = False
                            trunc_sum = body_content[:100] + ("..." if len(body_content) > 100 else "")
                            err_msg = f"Soft failure: Summary content overlap too low. Words shared: {sorted(overlap_words)}. Got: '{trunc_sum}'"
            
            throughput = len(summary_content) / elapsed if (passed and elapsed > 0) else 0
            
            results.append({
                "backend": name,
                "backend_type": "summarizer",
                "task_name": "summarize",
                "pass_fail": "pass" if passed else "fail",
                "load_time_sec": 0.0,  # run-to-completion loads model on run
                "latency_sec": elapsed,
                "throughput": throughput,
                "throughput_units": "chars/sec",
                "error": err_msg,
            })
        except Exception as e:
            results.append({
                "backend": name,
                "backend_type": "summarizer",
                "task_name": "summarize",
                "pass_fail": "fail",
                "load_time_sec": 0.0,
                "latency_sec": time.time() - start_time,
                "throughput": 0.0,
                "throughput_units": "chars/sec",
                "error": str(e),
            })
        return results

    # ─── Embedder Benchmarks ───────────────────────────────────────────────────

    def _benchmark_embedder(self, name: str, cfg: KnrsConfig) -> list[dict[str, Any]]:
        results = []
        cfg.embedder_name = name

        texts = [
            "Rome was founded in 753 BC by Romulus.",
            "Julius Caesar conquered Gaul and was assassinated in 44 BC.",
            "Augustus became the first Emperor of Rome.",
            "The Pax Romana was a long period of relative peace and stability.",
            "Rome defeated Carthage in the three Punic Wars.",
        ]

        logger.info("Benchmarking embedder '%s'...", name)
        
        # Measure startup/load time
        start_load = time.time()
        try:
            with EmbedderSession(cfg) as session:
                load_time = time.time() - start_load
                
                # Measure inference execution time
                start_exec = time.time()
                embeddings = session.embed(texts)
                exec_time = time.time() - start_exec
                
                # Validation
                passed = True
                err_msg = None
                if embeddings is None or len(embeddings) != len(texts):
                    passed = False
                    err_msg = f"Embeddings size mismatch: expected {len(texts)}, got {len(embeddings) if embeddings is not None else 0}"
                elif len(embeddings) > 0 and embeddings[0].shape[0] < 128:
                    passed = False
                    err_msg = f"Embedding dimensions too low: {embeddings[0].shape[0]}"
                
                throughput = len(texts) / exec_time if exec_time > 0 else 0
                
                results.append({
                    "backend": name,
                    "backend_type": "embedder",
                    "task_name": "embed_batch",
                    "pass_fail": "pass" if passed else "fail",
                    "load_time_sec": load_time,
                    "latency_sec": exec_time,
                    "throughput": throughput,
                    "throughput_units": "embeddings/sec",
                    "error": err_msg,
                })
        except Exception as e:
            results.append({
                "backend": name,
                "backend_type": "embedder",
                "task_name": "embed_batch",
                "pass_fail": "fail",
                "load_time_sec": time.time() - start_load,
                "latency_sec": 0.0,
                "throughput": 0.0,
                "throughput_units": "embeddings/sec",
                "error": str(e),
            })
        return results

    # ─── Agent Benchmarks ──────────────────────────────────────────────────────

    def _benchmark_agent(self, name: str, cfg: KnrsConfig) -> list[dict[str, Any]]:
        results = []
        cfg.agent_backend_name = name

        messages = [
            {"role": "user", "content": "Write a single sentence about Rome's Punic Wars."}
        ]

        logger.info("Benchmarking agent '%s'...", name)
        
        # Measure startup/load time
        start_load = time.time()
        try:
            with AgentSession(cfg) as session:
                load_time = time.time() - start_load
                
                # Measure generation execution time
                start_exec = time.time()
                # Use max_tokens = 100 to limit the test run
                response = session.generate(messages, max_tokens=100)
                exec_time = time.time() - start_exec
                
                # Validation
                passed = True
                err_msg = None
                response_clean = response.strip()
                
                if not response_clean:
                    passed = False
                    err_msg = "Agent returned an empty response."
                elif check_refusal_or_excuse(response_clean):
                    passed = False
                    err_msg = "Soft failure: Agent returned refusal/excuse template."
                
                # Estimate token throughput (approx. 4 characters per token)
                estimated_tokens = len(response_clean) / 4.0
                throughput = estimated_tokens / exec_time if exec_time > 0 else 0
                
                results.append({
                    "backend": name,
                    "backend_type": "agent",
                    "task_name": "chat_generation",
                    "pass_fail": "pass" if passed else "fail",
                    "load_time_sec": load_time,
                    "latency_sec": exec_time,
                    "throughput": throughput,
                    "throughput_units": "tokens/sec (estimated)",
                    "error": err_msg,
                })
        except Exception as e:
            results.append({
                "backend": name,
                "backend_type": "agent",
                "task_name": "chat_generation",
                "pass_fail": "fail",
                "load_time_sec": time.time() - start_load,
                "latency_sec": 0.0,
                "throughput": 0.0,
                "throughput_units": "tokens/sec (estimated)",
                "error": str(e),
            })
        return results

    # ─── File Operations & Formatting ──────────────────────────────────────────

    def _save_results(self, run_record: dict[str, Any]) -> None:
        """Save results list to benchmark_results.json, keeping the latest run per host."""
        # 1. Update the workspace results file
        history = []
        if self.results_path.exists():
            try:
                content = self.results_path.read_text(encoding="utf-8")
                if content.strip():
                    history = json.loads(content)
                    if not isinstance(history, list):
                        history = [history]
            except Exception as e:
                logger.error("Failed to read existing benchmark history: %s", e)

        history = merge_run_record(history, run_record)

        try:
            temp_path = self.results_path.with_suffix(".json.tmp")
            temp_path.write_text(json.dumps(history, indent=4), encoding="utf-8")
            temp_path.replace(self.results_path)
            logger.info("Results saved to: %s", self.results_path)
        except Exception as e:
            logger.error("Failed to save benchmark results: %s", e)

        # 2. Also update the user's config directory backup (keeping latest per host too)
        config_results_path = Path.home() / ".config" / "knrs" / "benchmark_runs.json"
        config_history = []
        if config_results_path.exists():
            try:
                content = config_results_path.read_text(encoding="utf-8")
                if content.strip():
                    config_history = json.loads(content)
                    if not isinstance(config_history, list):
                        config_history = [config_history]
            except Exception as e:
                logger.error("Failed to read global benchmark backup: %s", e)

        config_history = merge_run_record(config_history, run_record)

        try:
            config_results_path.parent.mkdir(parents=True, exist_ok=True)
            temp_config_path = config_results_path.with_suffix(".json.tmp")
            temp_config_path.write_text(json.dumps(config_history, indent=4), encoding="utf-8")
            temp_config_path.replace(config_results_path)
        except Exception as e:
            logger.error("Failed to save global benchmark backup: %s", e)
