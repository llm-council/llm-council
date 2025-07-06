"""Script to run a sample Language Model Council with specified evaluation type.

python scripts/update_example_council_run.py --eval_type pairwise

python scripts/update_example_council_run.py --eval_type rubric
"""

from lm_council.council import LanguageModelCouncil
from lm_council.judging import PRESET_EVAL_CONFIGS
from dotenv import load_dotenv
import asyncio
import argparse


load_dotenv()


async def main(eval_type: str):
    eval_config_key = (
        "default_pairwise" if eval_type == "pairwise" else "default_rubric"
    )
    lmc = LanguageModelCouncil(
        models=[
            "meta-llama/llama-3.1-8b-instruct",
            "deepseek/deepseek-r1-0528",
            "google/gemini-2.5-flash-lite-preview-06-17",
            "x-ai/grok-3-mini",
        ],
        judge_models=[
            "google/gemini-2.5-flash-preview-05-20",
            "x-ai/grok-3-mini",
            "deepseek/deepseek-r1-0528",
        ],
        eval_config=PRESET_EVAL_CONFIGS[eval_config_key],
    )

    completions, judgements = await lmc.execute(
        ["Say hello.", "Say goodbye.", "What is your name?"]
    )

    lmc.save(f"analysis/sample_council/{args.eval_type}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run LanguageModelCouncil with specified eval config."
    )
    parser.add_argument(
        "--eval_type",
        choices=["pairwise", "rubric"],
        default="pairwise",
        help="Evaluation type: 'pairwise' or 'rubric'",
    )
    args = parser.parse_args()

    asyncio.run(main(args.eval_type))
