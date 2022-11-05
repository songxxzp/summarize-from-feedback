#!/usr/bin/env python3
import sys, os
sys.path.append(os.getcwd())
import json
with open("./config.json") as f:
    DATA_PATH = json.load(f)["DATA_PATH"]

import fire

from summarize_from_feedback import eval_rm
from summarize_from_feedback.utils import experiment_helpers as utils
from summarize_from_feedback.utils.combos import combos, bind, bind_nested
from summarize_from_feedback.utils.experiments import experiment_def_launcher


def experiment_definitions():
    rm4 = combos(
        bind_nested("task", utils.tldr_task),
        bind("mpi", 1),
        bind_nested("reward_model_spec", utils.rm4()),
        bind("input_path", DATA_PATH + "/samples/sup4_ppo_rm4"),
    )

    test = combos(
        bind_nested("task", utils.test_task),
        bind_nested("reward_model_spec", utils.random_teeny_model_spec()),
        bind("mpi", 1),
        bind("input_path", DATA_PATH + "/samples/test"),
    )
    test_cpu = combos(
        test,
        bind_nested("reward_model_spec", utils.stub_model_spec()),
    )
    tldrtest = combos(
        bind_nested("task", utils.test_tldr_task),
        bind_nested("reward_model_spec", utils.random_teeny_model_spec(n_shards=2)),
        bind("mpi", 2),
    )
    return locals()


if __name__ == "__main__":
    fire.Fire(
        experiment_def_launcher(
            experiment_dict=experiment_definitions(), main_fn=eval_rm.main, mode="local"
        )
    )
