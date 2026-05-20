from __future__ import annotations

import math
import os
import time
import logging
from typing import Any, TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Dict

from .engine import BaseEngine
from .cache import WorkCache
from .markdown import get_answer_from_output

logger = logging.getLogger("summarizer_core.summarizer")

def chunked_summarize(engine: BaseEngine, content: str, filepath: str, chunk_size: int, doc_hash: str, chunk_sum_size: int = 500, final_sum_tokens: int = 1500) -> str:
    num_chunks: int = math.ceil(len(content) / chunk_size)
    filename: str = os.path.basename(filepath)
    
    if num_chunks == 0: 
        return ""
        
    if num_chunks == 1:
        logger.info(f"Document '{filename}' fits in one chunk. Summarizing directly...")
        prompt_text = f"The following is the full text of '{filepath}'. Please provide a detailed summary:\n\n{content}"
        
        prompt: str | list[dict[str, str]] = prompt_text
        if hasattr(engine, 'format_prompt'):
            formatted = engine.format_prompt([{"role": "user", "content": prompt_text}])
            if formatted:
                prompt = formatted
            
        output = engine.generate(prompt, max_tokens=final_sum_tokens)
        return get_answer_from_output(output)
        
    cache = WorkCache()
    chunk_summaries, start_index = cache.load_progress(doc_hash, chunk_size)
    
    if start_index > 0:
        if start_index >= num_chunks:
            logger.info(f"[{filename}] All chunks already processed. Resuming final consolidation...")
        else:
            logger.info(f"[{filename}] Resuming from chunk {start_index+1}/{num_chunks}...")
    else:
        logger.info(f"[{filename}] Starting summarization process, {num_chunks} chunks...")
  
    for i in range(start_index, num_chunks):
        start = i * chunk_size
        end = start + chunk_size
        chunk = content[start:end]
  
        if not chunk.strip():
            logger.info(f"Chunk {i+1} is empty or whitespace only, skipping.")
            cache.save_progress(doc_hash, chunk_size, chunk_summaries, i + 1, filepath)
            continue
  
        chunk_start = time.time()
        # logger.info(f"[{filename}] Summarizing chunk {i+1}/{num_chunks}...")
        
        prompt_text = f"Briefly summarize this part of document '{filepath}':\n\n{chunk}"
        
        chunk_prompt: str | list[dict[str, str]] = prompt_text
        if hasattr(engine, 'format_prompt'):
            formatted_chunk = engine.format_prompt([{"role": "user", "content": prompt_text}])
            if formatted_chunk:
                chunk_prompt = formatted_chunk
            
        output = engine.generate(chunk_prompt, max_tokens=chunk_sum_size)
        duration = time.time() - chunk_start
        logger.info(f"[{filename}] Finished chunk {i+1}/{num_chunks} in {duration:.1f}s")
        
        extracted_output = get_answer_from_output(output)
        chunk_summaries.append(extracted_output)
        cache.save_progress(doc_hash, chunk_size, chunk_summaries, i + 1, filepath)
  
    logger.info(f"[{filename}] Consolidating final summary...")
    
    if not chunk_summaries:
        logger.info(f"No valid summaries generated for {filename}, returning empty summary.")
        cache.clear_progress(doc_hash, chunk_size)
        return ""
        
    consolidated_text = "\n\n".join(chunk_summaries)
    
    final_prompt_text = f"The following are summaries of segments from '{filepath}'. Please combine them into a single coherent, detailed summary:\n\n{consolidated_text}"
    
    final_prompt: str | list[dict[str, str]] = final_prompt_text
    if hasattr(engine, 'format_prompt'):
        formatted_final = engine.format_prompt([{"role": "user", "content": final_prompt_text}])
        if formatted_final:
            final_prompt = formatted_final
        
    output = engine.generate(final_prompt, max_tokens=final_sum_tokens)
    final_summary = get_answer_from_output(output)
    
    cache.clear_progress(doc_hash, chunk_size)
    return final_summary
