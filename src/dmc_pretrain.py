# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import os

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from arguments import parser
from datamodules.dmc_datamodule import DMCMultiDomainDataModule
from datamodules.dummy_datamodule import RandomDataset
from models.multitask_ct_module import MultiTaskCTLitModule


def main(args):
    # set seed for reproducibility, although the trainer does not allow deterministic for this implementation
    pl.seed_everything(args.seed, workers=True)

    with open(args.multi_config) as jsonfile:
        args.source_envs = json.load(jsonfile)

    if args.source_data_type == "full":
        args.dataset_types = ["fullcollect"]
    elif args.source_data_type == "rand":
        args.dataset_types = ["randcollect"]
    elif args.source_data_type == "mix":
        args.dataset_types = ["fullcollect", "randcollect"]
        args.num_steps = args.num_steps // 2
    args.biased_multi = True

    # init data module
    dmc_data = DMCMultiDomainDataModule.from_argparse_args(args)

    # init training module
    dict_args = vars(args)
    if args.load_model_from:
        model = MultiTaskCTLitModule.load_from_checkpoint(args.load_model_from, **dict_args)
        print("loaded model from", args.load_model_from)
    else:
        model = MultiTaskCTLitModule(**dict_args)

    # init root dir
    os.makedirs(args.output_dir, exist_ok=True)
    print("output dir", args.output_dir)

    # checkpoint saving metrics
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(args.output_dir, "checkpoints"),
        # filename="best_model",
        filename="checkpoint_{epoch:02d}",
        mode="min",
        save_top_k=args.save_k,
        monitor="val/avg_loss",
        save_last=True,
    )

    logger = TensorBoardLogger(os.path.join(args.output_dir, "tb_logs"), name="train")

    # init trainer
    trainer = pl.Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        num_nodes=args.nodes,
        default_root_dir=args.output_dir,
        min_epochs=1,
        max_epochs=args.epochs,
        callbacks=[checkpoint_callback],
        strategy="ddp",
        fast_dev_run=False,
        logger=logger,
    )

    # start training
    trainer.fit(model, datamodule=dmc_data)

    # testing
    dummy = RandomDataset()
    trainer.test(model, dummy)


if __name__ == "__main__":
    parser = DMCMultiDomainDataModule.add_argparse_args(parser)
    parser = MultiTaskCTLitModule.add_model_specific_args(parser)

    args = parser.parse_args()

    main(args)
