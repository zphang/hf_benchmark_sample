import dataclasses
import tqdm.auto as tqdm

import torch

import datasets
import transformers

from hf_benchmark_sample.task_base import BaseTaskRunConfiguration, BaseTaskInterface


@dataclasses.dataclass
class RteTaskRunConfiguration(BaseTaskRunConfiguration):
    max_input_seq_length: int = 128
    max_num_examples: int = 16
    batch_size: int = 16


class RteTaskInterface(BaseTaskInterface):

    def prepare_model(self, run_config: RteTaskRunConfiguration):
        model = transformers.AutoModelForSequenceClassification.from_pretrained(
            run_config.model_name_or_path,
        )
        return model

    def prepare_tokenizer(self, run_config: RteTaskRunConfiguration):
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            run_config.tokenizer_name_or_path,
        )
        return tokenizer

    def prepare_dataset(
        self,
        tokenizer,
        run_config: RteTaskRunConfiguration
    ):
        def tokenize_fn(batch):
            tokenized = tokenizer(
                batch["sentence1"], batch["sentence2"],
                return_tensors="pt",
                padding="max_length",
                truncation="longest_first",
                max_length=run_config.max_input_seq_length,
            )
            tokenized["label"] = torch.LongTensor(batch["label"])
            return tokenized

        dataset = datasets.load_dataset("glue", name="rte")
        tokenized_dataset = dataset["validation"].map(
            tokenize_fn,
            batched=True,
        )
        return tokenized_dataset

    def compute_outputs(
        self,
        model,
        tokenizer,
        dataset,
        run_config: RteTaskRunConfiguration,
    ):
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=run_config.batch_size,
            collate_fn=transformers.default_data_collator,
        )
        logits = []
        for batch in tqdm.tqdm(dataloader):
            input_ids = batch["input_ids"].cuda()
            with torch.inference_mode():
                out = model(input_ids)
            logits.append(out.logits)
        return {
            "logits": torch.cat(logits),
        }

    def evaluate(self, model_outputs, tokenizer, dataset):
        labels = torch.LongTensor(dataset["label"])
        preds = model_outputs["logits"].argmax(-1).cpu()
        accuracy = (preds == labels).float().mean()
        return {
            "accuracy": accuracy.item(),
        }
