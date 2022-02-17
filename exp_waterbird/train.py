# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

#!/usr/bin/env python

import os
import sys
import json
import time
import torch
#import submitit
import argparse
import numpy as np
from tqdm import tqdm

import models
from datasets import get_loaders


class Tee:
    def __init__(self, fname, stream, mode="a+"):
        self.stream = stream
        self.file = open(fname, mode)

    def write(self, message):
        self.stream.write(message)
        self.file.write(message)
        self.flush()

    def flush(self):
        self.stream.flush()
        self.file.flush()

def randl(l_):
    return l_[torch.randperm(len(l_))[0]]

class Recorder():
    def __init__(self):
        self.values = dict()

    def save(self, key, val, i=None):
        if i is not None:
            val = (i, val)
        if key in self.values.keys():
            self.values[key].append(val)
        else:
            self.values[key] = [val]

    def get(self, key):
        return self.values[key]


def parse_args():
    parser = argparse.ArgumentParser(description='Balancing baselines')
    parser.add_argument('--output_dir', type=str, default='outputs')
    parser.add_argument('--slurm_output_dir', type=str, default='slurm_outputs')
    parser.add_argument('--data_path', type=str, default='data')
    parser.add_argument('--slurm_partition', type=str, default=None)
    parser.add_argument('--max_time', type=int, default=3*24*60)
    parser.add_argument('--num_hparams_seeds', type=int, default=20)
    parser.add_argument('--num_init_seeds', type=int, default=5)
    parser.add_argument('--selector', type=str, default='min_acc_va')
    return vars(parser.parse_args())


def run_experiment(args):
    recorder = Recorder()
    start_time = time.time()
    torch.manual_seed(args["init_seed"])
    np.random.seed(args["init_seed"])
    loaders = get_loaders(args["data_path"], args["dataset"], args["batch_size"], args["method"])

    sys.stdout = Tee(os.path.join(
        #args["output_dir"], 'seed_{}_{}.out'.format(
        #    args["hparams_seed"], args["init_seed"])), sys.stdout)
        args["output_dir"], 'alpha_{}.out'.format(
            args["alpha"])), sys.stdout)
    sys.stderr = Tee(os.path.join(
        args["output_dir"], 'alpha_{}.err'.format(
            args["alpha"])), sys.stderr)
    checkpoint_file = os.path.join(
        args["output_dir"], 'alpha_{}.pt'.format(
            args["alpha"]))
    best_checkpoint_file = os.path.join(
        args["output_dir"],
        "alpha_{}.best.pt".format(args["alpha"]),
    )
    result_file = os.path.join(
        args["output_dir"],
        "alpha_{}.results".format(args["alpha"]))

    model = {
        "erm": models.ERM,
        "suby": models.ERM,
        "subg": models.ERM,
        "rwy": models.ERM,
        "rwg": models.ERM,
        "dro": models.GroupDRO,
        "jtt": models.JTT
    }[args["method"]](args, loaders["tr"])

    last_epoch = 0
    best_selec_val = float('-inf')
    #if os.path.exists(checkpoint_file):
    #    model.load(checkpoint_file)
    #    last_epoch = model.last_epoch
    #    best_selec_val = model.best_selec_val
    print("Training...")
    for epoch in range(last_epoch, args["num_epochs"]):
        print('\nEpoch: %d' % epoch)
        if epoch == args["T"] + 1 and args["method"] == "jtt":
            loaders = get_loaders(
                args["data_path"],
                args["dataset"],
                args["batch_size"],
                args["method"],
                model.weights.tolist())

        for i, x, y, g in tqdm(loaders["tr"]):
            model.update(i, x, y, g, epoch)

        result = {
            "args": args, "epoch": epoch, "time": time.time() - start_time}
        print("monitoring...")
        for loader_name, loader in loaders.items():
            computer = model.computers(loader)
            avg_acc, group_accs = computer['accuracies']
            align_y, align_g, align_init = computer['alignments']
            result["acc_" + loader_name] = group_accs
            result["avg_acc_" + loader_name] = avg_acc
            result["aligny_" + loader_name] = align_y
            result["aligng_" + loader_name] = align_g
            result["aligninit_" + loader_name] = align_init
            recorder.save("acc_" + loader_name, group_accs)
            recorder.save("avg_acc_" + loader_name, avg_acc)
            recorder.save("aligny_" + loader_name, align_y)
            recorder.save("aligng_" + loader_name, align_g)
            recorder.save("aligninit_" + loader_name, align_init)
            recorder.save("activ_change_" + loader_name, activ_change)
            recorder.save("eff_rank_" + loader_name, eff_change)

        selec_value = {
            "min_acc_va": min(result["acc_va"]),
            "avg_acc_va": result["avg_acc_va"],
        }[args["selector"]]

        if selec_value >= best_selec_val:
            model.best_selec_val = selec_value
            best_selec_val = selec_value
            model.save(best_checkpoint_file)

        model.save(checkpoint_file)
        print(json.dumps(result))
        torch.save(recorder, result_file)
    return recorder


if __name__ == "__main__":
    args = parse_args()

    commands = []
    for hparams_seed in range(args["num_hparams_seeds"]):
        torch.manual_seed(hparams_seed)
        args["hparams_seed"] = hparams_seed

        args["dataset"] = randl(["waterbirds"])
            #["waterbirds", "celeba", "multinli", "civilcomments"])

        args["method"] = randl(["erm"])
            #["erm", "suby", "subg", "rwy", "rwg", "dro", "jtt"])

        args["num_epochs"] = {
            "waterbirds": 300 + 60,
            "celeba": 50 + 10,
            "multinli": 5 + 2,
            "civilcomments": 5 + 2
        }[args["dataset"]]

        args["eta"] = 0.1
        args["lr"] = randl([1e-5, 1e-4, 1e-3])
        args["weight_decay"] = randl([1e-4, 1e-3, 1e-2, 1e-1, 1])

        if args["dataset"] in ["waterbirds", "celeba"]:
            args["batch_size"] = randl([2, 4, 8, 16, 32, 64, 128])
        else:
            args["batch_size"] = randl([2, 4, 8, 16, 32])

        args["up"] = randl([4, 5, 6, 20, 50, 100])
        args["T"] = {
            "waterbirds": randl([40, 50, 60]),
            "celeba": randl([1, 5, 10]),
            "multinli": randl([1, 2]),
            "civilcomments": randl([1, 2])
        }[args["dataset"]]

        for init_seed in range(args["num_init_seeds"]):
            args["init_seed"] = init_seed
            commands.append(dict(args))

    os.makedirs(args["output_dir"], exist_ok=True)
    torch.manual_seed(0)
    commands = [commands[int(p)] for p in torch.randperm(len(commands))]

    if args['slurm_partition'] is not None:
        executor = submitit.SlurmExecutor(folder=args['slurm_output_dir'])
        executor.update_parameters(
            time=args["max_time"],
            gpus_per_node=1,
            array_parallelism=512,
            cpus_per_task=4,
            partition=args["slurm_partition"])
        executor.map_array(run_experiment, commands)
    else:
        for command in commands:
            run_experiment(command)
