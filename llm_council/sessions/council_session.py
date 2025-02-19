import pandas as pd
import os
import json
import jsonlines
from pathlib import Path
from llm_council.judging.schema import EvaluationConfig, save_config, load_config


class CouncilSession:
    """Contains everything related to a single LMC execution."""

    # LLM, prompt, completion text, completion object, input tokens used, output tokens used, provider
    completions: pd.DataFrame

    # LLM_judge, llm_a, llm_b, rating, judgment_object, input tokens used, output tokens used, provider
    judgments: pd.DataFrame

    # Prompt used for this session.
    completion_prompt: str

    # Map of llm short name -> fully qualified LLM
    llms: dict[str, str]

    # LLMs that were judging. If None, then all llms were also involved in judging.
    judging_llms: list[str] | None

    # Evaluation configuration for this session.
    evaluation_config: EvaluationConfig

    def __init__(
        self,
        llms,
        prompt,
        completions_df,
        judging_df,
        evaluation_config,
    ):
        self.llms = llms
        self.prompt = prompt
        self.completions_df = completions_df
        self.judging_df = judging_df
        self.evaluation_config = evaluation_config

    def describe(self):
        pass

    def rankings(self):
        pass

    # Analysis functions.
    def win_rate_heatmap(self):
        pass

    def win_rate_df(self):
        pass

    def affinity_heatmap(self):
        pass

    def affinity_df(self):
        pass

    def agreement_heatmap(self):
        pass

    def agreement_df(self):
        pass

    def length_bias(self):
        pass

    def self_enhancement_bias(self):
        pass

    def usage(self):
        pass

    def save(self, outdir):
        # Save all artifacts to a directory.
        """
        outdir/
        - llm_to_provider_map.json
        - prompt.txt
        - completions.jsonl
        - judging.jsonl
        - evaluation_config.json
        """
        os.makedirs(outdir, exist_ok=True)
        with open(os.path.join(outdir, "llms.json"), "w") as f:
            json.dump(self.llms, f)
        with open(os.path.join(outdir, "prompt.txt"), "w") as f:
            f.write(self.prompt)
        self.completions_df.to_json(
            os.path.join(outdir, "completions.jsonl"), orient="records", lines=True
        )
        self.judging_df.to_json(
            os.path.join(outdir, "judging.jsonl"), orient="records", lines=True
        )
        self.completions_df.to_csv(os.path.join(outdir, "completions.csv"), index=False)
        self.judging_df.to_csv(os.path.join(outdir, "judging.csv"), index=False)
        save_config(self.evaluation_config, "evaluation_config.json")

    @staticmethod
    def load(outdir):
        # Load all artifacts from a directory.
        with open(os.path.join(outdir, "llms.json"), "r") as f:
            llms = json.load(f)
        with open(os.path.join(outdir, "prompt.txt"), "r") as f:
            prompt = f.read()
        completions_df = pd.read_json(
            os.path.join(outdir, "completions.jsonl"), orient="records", lines=True
        )
        judging_df = pd.read_json(
            os.path.join(outdir, "judging.jsonl"), orient="records", lines=True
        )
        completions_df.to_csv(os.path.join(outdir, "completions.csv"), index=False)
        judging_df.to_csv(os.path.join(outdir, "judging.csv"), index=False)
        evaluation_config = load_config("evaluation_config.json")

        return CouncilSession(
            llms=llms,
            prompt=prompt,
            completions_df=judging_df,
            judging_df=judging_df,
            evaluation_config=evaluation_config,
        )
