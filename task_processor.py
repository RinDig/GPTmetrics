import asyncio
import logging
from tqdm.notebook import tqdm 

from llm_interface.api_clients import call_model_api, cost_tracker
from configs.app_config import MAX_CONCURRENT_CALLS

logger = logging.getLogger(__name__)

async def process_tasks_in_chunks(task_list, chunk_size=MAX_CONCURRENT_CALLS):
    results = []
    total_tasks = len(task_list)
    
    openai_queue = []
    anthropic_queue = []
    llama_queue = []
    
    for task in task_list:
        if task["model_name"] in ["OpenAI", "Grok"]:
            openai_queue.append(task)
        elif task["model_name"] == "Claude":
            anthropic_queue.append(task)
        else: 
            llama_queue.append(task)
    
    pbar = tqdm(total=total_tasks, desc="Processing tasks")
    
    async def process_queue(queue, semaphore, rate_limit):
        queue_results = []
        for i in range(0, len(queue), chunk_size):
            chunk = queue[i:i + chunk_size]
            async with semaphore:
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
                pbar.update(len(chunk))
                await asyncio.sleep(rate_limit)
        return queue_results

    openai_sem = asyncio.Semaphore(3) 
    anthropic_sem = asyncio.Semaphore(5)
    llama_sem = asyncio.Semaphore(10) 
    
    queue_tasks = [
        process_queue(openai_queue, openai_sem, 1.0),
        process_queue(anthropic_queue, anthropic_sem, 0.5),
        process_queue(llama_queue, llama_sem, 0.2),
    ]
    
    all_results_processed = await asyncio.gather(*queue_tasks)
    pbar.close()
    
    for queue_result_list in all_results_processed:
        for task_info, result_data in queue_result_list:
            if isinstance(result_data, Exception):
                logger.error(f"Task failed: {str(result_data)}")
                min_scale, max_scale = task_info["scale_range"]
                midpoint = (min_scale + max_scale) / 2
                result_dict = {
                    "model_name": task_info["model_name"],
                    "numeric_score": midpoint,
                    "label": None,
                    "justification": f"ERROR: {str(result_data)}",
                    "raw_response": str(result_data),
                    "duration": None
                }
            else:
                result_dict = result_data
            
            min_scale, max_scale = task_info["scale_range"]
            if not (min_scale <= result_dict["numeric_score"] <= max_scale):
                logger.warning(f"Score {result_dict['numeric_score']} out of range for {task_info['question_id']}, using midpoint")
                result_dict["numeric_score"] = (min_scale + max_scale) / 2
                result_dict["justification"] = f"RANGE ERROR: Original score: {result_dict['numeric_score']}"
            
            # Merge task_info into the result_dict, prioritizing keys from result_dict in case of overlap
            # except for 'cost_tracker' which is handled separately if needed.
            final_result_entry = {**task_info, **result_dict}
            results.append(final_result_entry)
    
    logger.info(f"Processed {len(results)}/{total_tasks} tasks. Current token usage: {dict(cost_tracker)}")
    return results 