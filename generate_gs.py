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
    perceptual_dimensions,
    vocab_size,
    n_distractors,
    n_epoch,
    max_len,
    train_samples,
    validation_samples,
    test_samples,
    sender_lr,
    receiver_lr,
    batch_size,
):
    out = f"""python -m egg.zoo.objects_game.train \
    --experiment_id {experiment_id} \
    --perceptual_dimensions {perceptual_dimensions} \
    --vocab_size {vocab_size} \
    --n_distractors {n_distractors} \
    --n_epoch {n_epoch} \
    --max_len {max_len}  \
    --sender_lr {sender_lr} \
    --receiver_lr {receiver_lr} \
    --batch_size {batch_size} \
    --random_seed {seed} \
    --data_seed {seed} \
    --train_samples {train_samples} \
    --validation_samples {validation_samples} \
    --test_samples {test_samples} \
    --evaluate \
    --dump_msg_folder '../messages' \
    --shuffle_train_data \
    --mode {mode} &
"""
    return out


def main():
    experiment_name = "gs"
    options = {
        "perceptual_dimensions": ["[10,10]"],
        "vocab_size": [20],
        "n_distractors": [9],
        "n_epoch": [5000],
        "max_len": [2],
        "train_samples": [80],
        "validation_samples": [10],
        "test_samples": [10],
        "sender_lr": [0.001, 0.0001, 0.0005],
        "receiver_lr": [0.001, 0.0001, 0.0005],
        "batch_size": [5],
    }
    options = list(itertools.product(*options.values()))
    samples = options
    seeds = [0]
    modes = ["gs", "gs-hard",]

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