import itertools
import os
import math
import random

PROCESS_PER_SCRIPT = 1

def template_file(texts):
    text = "".join(texts)
    out = f"""#!/bin/bash
#$ -cwd
#$ -e ./logs/
#$ -o ./logs/
#$ -l vf=4G

source /data/nlp/lunar_pilot_env/bin/activate
echo 'Starting job'
mkdir -p data
mkdir -p output

mkdir -p ./logs/
mkdir -p ./output/

{text}

wait
"""
    return out


def template_exp_option(
    experiment_id,
    mode,
    seed,
    vocab_size,
    n_distractors,
    n_epoch,
    max_len,
    lr,
    sender_entropy_coeff,
    receiver_entropy_coeff,
    batch_size,
):
    sender_lr = receiver_lr = lr
    out = f"""python -m egg.zoo.objects_game_concepts.train \
    --experiment_id {experiment_id} \
    --vocab_size {vocab_size} \
    --n_distractors {n_distractors} \
    --n_epoch {n_epoch} \
    --max_len {max_len}  \
    --sender_lr {sender_lr} \
    --receiver_lr {receiver_lr} \
    --batch_size {batch_size} \
    --sender_entropy_coeff {sender_entropy_coeff} \
    --receiver_entropy_coeff {receiver_entropy_coeff} \
    --random_seed {seed} \
    --data_seed {seed} \
    --evaluate \
    --shuffle_train_data \
    --mode {mode} &
"""
    return out


def main():
    experiment_name = "a2c"
    options = {
        "vocab_size": [100],
        "n_distractors": [4],
        "n_epoch": [5_000],
        "max_len": [2, 5, 10],
        "lr": [0.0001], # [0.001, 0.0001, 0.0005],
        "sender_entropy_coeff": [0.01],
        "receiver_entropy_coeff": [0.001],
        "batch_size": [32],
    }
    options = list(itertools.product(*options.values()))
    samples = options
    seeds = [0, 1, 2]
    modes = ["rf", "rf-deterministic", "gs", "gs-hard"]

    exp_templates = []
    for mode in modes:
        for seed in seeds:
            for option in samples:
                option_str = "-".join([str(p) for p in option])
                experiment_id = f"{experiment_name}-{mode}-{seed}-{option_str}"
                template = template_exp_option(experiment_id, mode, seed, *option)
                exp_templates.append(template)

    exp_files = []
    for start_idx in range(0, len(exp_templates), PROCESS_PER_SCRIPT):
        batch = exp_templates[start_idx : start_idx + PROCESS_PER_SCRIPT]
        file = template_file(batch)
        exp_files.append(file)

    if not os.path.exists("./jobs"):
        os.makedirs("./jobs")

    for i, file in enumerate(exp_files):
        with open(f"./jobs/{experiment_name}_exp_{i}.sh", "w") as f:
            f.write(file)
    all_exp_file = "\n".join(
        [f"qsub ./jobs/{experiment_name}_exp_{i}.sh".format(i) for i in range(len(exp_files))]
    )
    with open(f"./jobs/{experiment_name}_all_exp.sh", "w") as f:
        f.write(all_exp_file)

if __name__ == "__main__":
    main()