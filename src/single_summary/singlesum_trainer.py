import argparse
import os
import random
import time

import numpy as np
import pytorch_lightning as pl
import torch

from src.single_summary.singlesum import Summarization


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.gpus > 0:
        torch.cuda.manual_seed_all(args.seed)


def add_generic_args(parser):
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )

    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
        "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument("--fast_dev_run", action="store_true")
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--n_tpu_cores", type=int, default=0)
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_predict", action="store_true", help="Whether to run predictions on the test set.")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=10,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--overwrite_cache", action="store_true")
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--val_check_interval", default=1.0, type=float)


def main(args):
    set_seed(args)

    if args.output_dir is None:
        args.output_dir = os.path.join("./results", f"{args.task}_{args.model_type}_{time.strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(args.output_dir)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        filepath=args.output_dir, monitor="val_loss", prefix='checkpoint',
        verbose=True, mode="min", save_top_k=3)

    model = Summarization(args)

    train_params = {}
    if args.fp16:
        train_params["use_amp"] = args.fp16
        train_params["amp_level"] = args.fp16_opt_level

    if args.gpus > 1:
        train_params["distributed_backend"] = "dp"

    trainer = pl.Trainer(
        accumulate_grad_batches=args.gradient_accumulation_steps,
        gpus=args.gpus,
        max_epochs=args.num_train_epochs,
        gradient_clip_val=args.max_grad_norm,
        checkpoint_callback=checkpoint_callback,
        fast_dev_run=args.fast_dev_run,
        val_check_interval=args.val_check_interval,
        weights_summary=None,
        resume_from_checkpoint=args.resume_from_checkpoint,
        **train_params,
    )

    if args.do_train:
        trainer.fit(model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    add_generic_args(parser)
    parser = Summarization.add_model_specific_args(parser)
    args = parser.parse_args()

    main(args)

