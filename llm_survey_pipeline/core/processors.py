"""Task processing and execution for LLM Survey Pipeline"""
import asyncio
import logging
import pandas as pd
from typing import List, Dict
from tqdm import tqdm

from llm_survey_pipeline.core.api_clients import call_model_api
from llm_survey_pipeline.utils.cost_tracking import cost_tracker

logger = logging.getLogger(__name__)


async def process_tasks_in_chunks(task_list, chunk_size=5):
    """Process tasks with improved concurrency and rate limiting"""
    results = []
    total_tasks = len(task_list)
    
    # Create separate queues for different API providers to avoid rate limits
    openai_queue = []
    anthropic_queue = []
    llama_queue = []
    
    # Sort tasks by API provider
    for task in task_list:
        if task["model_name"] in ["OpenAI", "Grok"]:
            openai_queue.append(task)
        elif task["model_name"] == "Claude":
            anthropic_queue.append(task)
        else:  # Llama and DeepSeek
            llama_queue.append(task)
    
    # Create progress bar
    pbar = tqdm(total=total_tasks, desc="Processing tasks")
    
    async def process_queue(queue, semaphore, rate_limit):
        """Process a queue with rate limiting"""
        queue_results = []
        for i in range(0, len(queue), chunk_size):
            chunk = queue[i:i + chunk_size]
            async with semaphore:
                # Process chunk with rate limiting
                coros = [
                    call_model_api(
                        t["model_name"],
                        t["question_text"],
                        t["prompt_style"],
                        t["scale_range"],
                        0.0
                    )
                    for t in chunk
                ]
                chunk_results = await asyncio.gather(*coros, return_exceptions=True)
                queue_results.extend(zip(chunk, chunk_results))
                
                # Update progress bar
                pbar.update(len(chunk))
                
                # Rate limiting delay
                await asyncio.sleep(rate_limit)
        
        return queue_results

    # Create separate semaphores for each API provider
    openai_sem = asyncio.Semaphore(3)    # Allow 3 concurrent OpenAI calls
    anthropic_sem = asyncio.Semaphore(5)  # Allow 5 concurrent Anthropic calls
    llama_sem = asyncio.Semaphore(10)     # Allow 10 concurrent Llama calls
    
    # Process all queues concurrently with different rate limits
    queue_tasks = [
        process_queue(openai_queue, openai_sem, 1.0),      # 1 second between chunks
        process_queue(anthropic_queue, anthropic_sem, 0.5), # 0.5 seconds between chunks
        process_queue(llama_queue, llama_sem, 0.2),        # 0.2 seconds between chunks
    ]
    
    all_results = await asyncio.gather(*queue_tasks)
    
    # Close progress bar
    pbar.close()
    
    # Combine and process results
    for queue_result in all_results:
        for task_info, result in queue_result:
            if isinstance(result, Exception):
                logger.error(f"Task failed: {str(result)}")
                # Use scale midpoint for failed tasks
                min_scale, max_scale = task_info["scale_range"]
                midpoint = (min_scale + max_scale) / 2
                result_dict = {
                    "model_name": task_info["model_name"],
                    "numeric_score": midpoint,
                    "label": None,
                    "justification": f"ERROR: {str(result)}",
                    "raw_response": str(result),
                    "duration": None
                }
            else:
                result_dict = result
            
            # Validate numeric score
            min_scale, max_scale = task_info["scale_range"]
            if not (min_scale <= result_dict["numeric_score"] <= max_scale):
                logger.warning(f"Score {result_dict['numeric_score']} out of range for {task_info['question_id']}, using midpoint")
                result_dict["numeric_score"] = (min_scale + max_scale) / 2
                result_dict["justification"] = f"RANGE ERROR: Original score: {result_dict['numeric_score']}"
            
            result_dict.update(task_info)
            results.append(result_dict)
    
    logger.info(f"Processed {len(results)}/{total_tasks} tasks. Current token usage: {dict(cost_tracker)}")
    
    return results


def apply_reverse_score(row):
    """
    Apply reverse scoring for different scales.
    - MFQ scale is 1-5; reversed item => 6 - original
    - LWA and RWA scale is 1-7; reversed item => 8 - original
    """
    score = row["numeric_score"]
    reverse_flag = row.get("reverse_score", False)
    scale_name = row.get("scale_name", "")

    if pd.isna(score):
        return score  # No change if score is NaN

    if not reverse_flag:
        return score  # Return as-is if not a reverse-scored item

    # Handle each scale's reversing logic
    if scale_name == "MFQ":
        return 6 - score
    elif scale_name == "RWA":
        return 8 - score  # Same logic as LWA for 1-7 scale
    elif scale_name == "RWA2":
        return 8 - score  # Same logic as LWA for 1-7 scale
    elif scale_name == "LWA":
        return 8 - score
    else:
        # If any future scale needs reversing, define it here
        return score