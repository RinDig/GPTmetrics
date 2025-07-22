"""Main CLI interface for LLM Survey Pipeline"""
import asyncio
import pandas as pd
import logging
import os
from pathlib import Path

from llm_survey_pipeline.config import all_questions, all_scales
from llm_survey_pipeline.core import process_tasks_in_chunks, apply_reverse_score
from llm_survey_pipeline.utils import cost_tracker, save_refusal_responses, calculate_mfq_scores

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SurveyPipeline:
    def __init__(self, 
                 scales_to_run=None,
                 prompt_styles_to_run=None,
                 models_to_run=None,
                 num_calls_test=1,
                 temperature=0.0,
                 max_concurrent_calls=5,
                 output_dir="data/outputs"):
        
        self.scales_to_run = scales_to_run or ["RWA", "LWA", "RWA2"]
        self.prompt_styles_to_run = prompt_styles_to_run or ["minimal", "extreme_liberal", "extreme_conservitive"]
        self.models_to_run = models_to_run or ["OpenAI", "Claude", "Grok"]
        self.num_calls_test = num_calls_test
        self.temperature = temperature
        self.max_concurrent_calls = max_concurrent_calls
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def build_tasks(self):
        """Build the task list based on configuration"""
        tasks = []
        for q in all_questions:
            if q["scale_name"] not in self.scales_to_run:
                continue
            scale_name = q["scale_name"]
            question_id = q["id"]
            question_text = q["text"]
            scale_range = q["scale_range"]
            reverse_score = q.get("reverse_score", False)

            for model_name in self.models_to_run:
                for prompt_style in self.prompt_styles_to_run:
                    for run in range(1, self.num_calls_test + 1):
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
    
    async def run_survey(self):
        """Run the survey pipeline"""
        # Build tasks
        tasks = self.build_tasks()
        logger.info(f"Created {len(tasks)} tasks")
        
        # Process tasks
        results = await process_tasks_in_chunks(tasks, chunk_size=self.max_concurrent_calls)
        
        # Build DataFrame
        df_results = pd.DataFrame(results)
        
        # Apply reverse scoring
        df_results["scored_value"] = df_results.apply(apply_reverse_score, axis=1)
        
        # Save results
        output_path = self.output_dir / "unified_responses.csv"
        df_results.to_csv(output_path, index=False)
        logger.info(f"Saved responses to {output_path}")
        
        # Print summary
        print("\nProcessing Summary:")
        print(f"Total responses: {len(df_results)}")
        print(f"Token usage by model: {dict(cost_tracker)}")
        resp_rate = df_results['numeric_score'].notna().sum() / len(df_results) * 100
        print(f"Response rate: {resp_rate:.1f}%")
        
        # Save refusal responses
        refusal_path = self.output_dir / "refusal_responses.csv"
        save_refusal_responses(df_results, output_file=str(refusal_path))
        
        # Calculate MFQ scores if MFQ was run
        if "MFQ" in self.scales_to_run:
            calculate_mfq_scores(str(output_path))
        
        return df_results


def main():
    """Main entry point for CLI"""
    import argparse
    
    parser = argparse.ArgumentParser(description="LLM Survey Pipeline")
    parser.add_argument("--scales", nargs="+", default=["RWA", "LWA", "RWA2"],
                        help="Scales to run (e.g., RWA LWA MFQ NFC)")
    parser.add_argument("--prompts", nargs="+", 
                        default=["minimal", "extreme_liberal", "extreme_conservitive"],
                        help="Prompt styles to use")
    parser.add_argument("--models", nargs="+", default=["OpenAI", "Claude", "Grok"],
                        help="Models to test")
    parser.add_argument("--runs", type=int, default=1,
                        help="Number of runs per question")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Temperature for model responses")
    parser.add_argument("--output-dir", default="data/outputs",
                        help="Output directory for results")
    
    args = parser.parse_args()
    
    pipeline = SurveyPipeline(
        scales_to_run=args.scales,
        prompt_styles_to_run=args.prompts,
        models_to_run=args.models,
        num_calls_test=args.runs,
        temperature=args.temperature,
        output_dir=args.output_dir
    )
    
    # Run the pipeline
    asyncio.run(pipeline.run_survey())


if __name__ == "__main__":
    main()