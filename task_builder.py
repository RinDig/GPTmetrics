from scales.scale_definitions import ALL_QUESTIONS
from configs.app_config import SCALES_TO_RUN, MODELS_TO_RUN, PROMPT_STYLES_TO_RUN, NUM_CALLS_TEST

def build_tasks():
    tasks = []
    for q in ALL_QUESTIONS:
        if q["scale_name"] not in SCALES_TO_RUN:
            continue
        scale_name = q["scale_name"]
        question_id = q["id"]
        question_text = q["text"]
        scale_range = q["scale_range"]
        reverse_score = q.get("reverse_score", False)

        for model_name in MODELS_TO_RUN:
            for prompt_style in PROMPT_STYLES_TO_RUN:
                for run in range(1, NUM_CALLS_TEST + 1):
                    tasks.append({
                        "model_name": model_name,
                        "scale_name": scale_name,
                        "question_id": question_id,
                        "question_text": question_text,
                        "prompt_style": prompt_style,
                        "run_number": run,
                        "scale_range": scale_range,
                        "reverse_score": reverse_score,
                    })
    return tasks 