import asyncio
import logging
from tqdm.notebook import tqdm
from typing import List, Dict, Optional, Tuple

from llm_services import call_model_api, cost_tracker
from config import MODEL_CONFIG

# Set up logging
logger = logging.getLogger(__name__)

async def process_tasks_in_chunks(task_list: List[Dict], chunk_size: int = 5) -> List[Dict]:
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
    
    async def process_queue(queue: List[Dict], semaphore: asyncio.Semaphore, rate_limit: float) -> List[Tuple[Dict, Dict]]:
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
                        0.0  # Assuming temperature is fixed at 0.0 as per notebook
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
    # These values are taken from the notebook cell 8
    openai_sem = asyncio.Semaphore(3)    # Allow 3 concurrent OpenAI calls
    anthropic_sem = asyncio.Semaphore(5)  # Allow 5 concurrent Anthropic calls
    llama_sem = asyncio.Semaphore(10)     # Allow 10 concurrent Llama calls
    
    # Process all queues concurrently with different rate limits
    queue_tasks = [
        process_queue(openai_queue, openai_sem, 1.0),      # 1 second between chunks
        process_queue(anthropic_queue, anthropic_sem, 0.5), # 0.5 seconds between chunks
        process_queue(llama_queue, llama_sem, 0.2),        # 0.2 seconds between chunks
    ]
    
    all_raw_results = await asyncio.gather(*queue_tasks)
    
    # Close progress bar
    pbar.close()
    
    # Combine and process results
    for queue_result_list in all_raw_results:
        for task_info, result_data in queue_result_list:
            if isinstance(result_data, Exception):
                logger.error(f"Task failed: {str(result_data)} for question_id: {task_info.get('question_id', 'N/A')} with model: {task_info.get('model_name', 'N/A')}")
                # Use scale midpoint for failed tasks
                min_scale, max_scale = task_info["scale_range"]
                midpoint = (min_scale + max_scale) / 2
                processed_result = {
                    "model_name": task_info["model_name"],
                    "numeric_score": midpoint,
                    "label": None,
                    "justification": f"ERROR: {str(result_data)}",
                    "raw_response": str(result_data),
                    "duration": None
                }
            else:
                processed_result = result_data
            
            # Validate numeric score
            min_scale, max_scale = task_info["scale_range"]
            if not (min_scale <= processed_result["numeric_score"] <= max_scale):
                logger.warning(f"Score {processed_result['numeric_score']} out of range for {task_info['question_id']}, using midpoint")
                processed_result["numeric_score"] = (min_scale + max_scale) / 2
                processed_result["justification"] = f"RANGE ERROR: Original score: {processed_result['numeric_score']}"
            
            # Combine original task info with the processed result from the API call
            final_result_entry = {**task_info, **processed_result}
            results.append(final_result_entry)
    
    logger.info(f"Processed {len(results)}/{total_tasks} tasks. Current token usage: {dict(cost_tracker)}")
    
    return results
