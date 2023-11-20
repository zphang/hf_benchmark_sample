import torch

import hf_benchmark_sample.tasks.sample_rte_task as sample_rte_task


def main():
    # Create task config
    run_config = sample_rte_task.RteTaskRunConfiguration(
        model_name_or_path="TitanML/Roberta-Large-RTE",
        device=torch.device("cuda:0"),
    )

    # Create task object
    rte_task = sample_rte_task.RteTaskInterface()

    # Run and evaluate end-to-end
    eval_outputs = rte_task.run_end_to_end(run_config)
    print(eval_outputs["eval_outputs"]["accuracy"])

    # Upload results to a (pre-made) repo
    rte_task.upload(
        eval_outputs=eval_outputs,
        repo_id="zphang/hf_benchmark_sample",
        name="sample_rte_score.json"
    )


if __name__ == "__main__":
    main()
