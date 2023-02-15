import csv
import os
import time

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from arguments import parser
from datamodules.dmc_datamodule import DMCBCDataModule, DMCDataModule
from datamodules.dummy_datamodule import RandomDataset
from models.ct_module import CTLitModule


def main(args):

    # set seed for reproducibility, although the trainer does not allow deterministic for this implementation
    pl.seed_everything(args.seed, workers=True)

    bc = (args.model_type == "naive")
    # init data module
    if bc:
        dmc_data = DMCBCDataModule.from_argparse_args(args)
    else:
        dmc_data = DMCDataModule.from_argparse_args(args)

    # init training module
    dict_args = vars(args)

    model = CTLitModule(**dict_args)
    if args.load_model_from:
        model.load_my_checkpoint(
            args.load_model_from,
            no_action=args.no_load_action,
            strict=not args.no_strict,
            no_action_head=args.no_action_head,
        )
        print("loaded model from", args.load_model_from)

    # init root dir
    os.makedirs(args.output_dir, exist_ok=True)
    print("output dir", args.output_dir)

    # checkpoint saving metrics
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(args.output_dir, "checkpoints_bc"),
        filename="best_reward_model",
        mode="max",
        save_top_k=args.save_k,
        monitor="val/interactive_reward",
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
        # strategy='ddp_find_unused_parameters_false',
        fast_dev_run=False,
        logger=logger,
    )

    trainer.fit(model, datamodule=dmc_data)

    # testing
    dummy = RandomDataset()
    trainer.test(model, dummy, ckpt_path="best")


if __name__ == "__main__":
    parser = DMCDataModule.add_argparse_args(parser)
    parser = CTLitModule.add_model_specific_args(parser)

    args = parser.parse_args()

    main(args)
