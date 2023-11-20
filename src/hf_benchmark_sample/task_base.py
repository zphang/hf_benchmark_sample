import dataclasses
import json
import tempfile
from typing import Optional

import torch

import transformers
from huggingface_hub import login, create_repo, HfApi


@dataclasses.dataclass
class BaseTaskRunConfiguration:
    model_name_or_path: str
    tokenizer_name_or_path: Optional[str] = None
    device: torch.device = None

    def __post_init__(self):
        if self.tokenizer_name_or_path is None:
            self.tokenizer_name_or_path = self.model_name_or_path


class BaseTaskInterface:

    def prepare_model(self, run_config: BaseTaskRunConfiguration):
        raise NotImplementedError()

    def prepare_tokenizer(self, run_config: BaseTaskRunConfiguration):
        raise NotImplementedError()

    def prepare_dataset(
        self,
        tokenizer,
        run_config: BaseTaskRunConfiguration
    ):
        raise NotImplementedError()

    def compute_outputs(
        self,
        model,
        tokenizer,
        dataset,
        run_config: BaseTaskRunConfiguration,
    ):
        raise NotImplementedError()

    def evaluate(self, model_outputs, tokenizer, dataset):
        raise NotImplementedError()

    def format_for_results_for_upload(self, eval_outputs) -> dict:
        raise NotImplementedError()

    def run_end_to_end(self, run_config: BaseTaskRunConfiguration):
        model = self.prepare_model(run_config)
        model = model.to(run_config.device)
        tokenizer = self.prepare_tokenizer(run_config)
        dataset = self.prepare_dataset(
            tokenizer=tokenizer,
            run_config=run_config,
        )
        model_outputs = self.compute_outputs(
            model=model,
            tokenizer=tokenizer,
            dataset=dataset,
            run_config=run_config,
        )
        eval_outputs = self.evaluate(
            model_outputs=model_outputs,
            tokenizer=tokenizer,
            dataset=dataset,
        )
        return {
            "eval_outputs": eval_outputs,
            "model_outputs": model_outputs,
            "dataset": dataset,
            "tokenizer": tokenizer,
            "model": model,
        }

    def upload(self, eval_outputs, repo_id, name):
        # TODO: Check that repo exists
        api = HfApi()
        with tempfile.NamedTemporaryFile("w", delete=False) as f:
            f.write(json.dumps(
                self.format_for_results_for_upload(eval_outputs["eval_outputs"])
            ))
        api.upload_file(
            path_or_fileobj=f.name,
            path_in_repo=f"{name}.json",
            repo_id=repo_id,
            repo_type="dataset",
        )
