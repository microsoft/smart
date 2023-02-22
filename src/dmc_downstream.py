# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os

from omegaconf import OmegaConf

from datamodules.dmc_datamodule import DMCBCDataModule, DMCDataModule
from datamodules.dummy_datamodule import RandomDataset
from utils import pl_utils


def main(cfg):
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    if cfg.model.model_type == "naive":
        # bc = True
        dmc_data = DMCBCDataModule(**cfg.data)
    else:
        dmc_data = DMCDataModule(**cfg.data)

    model = pl_utils.instantiate_class(cfg["model"])
    if cfg.load_model_from:
        model.load_my_checkpoint(
            cfg.load_model_from,
            no_action=cfg.no_load_action,
            strict=not cfg.no_strict,
            no_action_head=cfg.no_action_head,
        )
        print("loaded model from", cfg.load_model_from)

    # init root dir
    os.makedirs(cfg.output_dir, exist_ok=True)
    print("output dir", cfg.output_dir)

    trainer = pl_utils.instantiate_trainer(cfg_dict)

    trainer.fit(model, datamodule=dmc_data)

    # testing
    dummy = RandomDataset()
    trainer.test(model, dummy, ckpt_path="best")


if __name__ == "__main__":
    cfg = OmegaConf.from_cli()

    if "base" in cfg:
        basecfg = OmegaConf.load(cfg.base)
        del cfg.base
        cfg = OmegaConf.merge(basecfg, cfg)
        print(OmegaConf.to_yaml(cfg))
        main(cfg)
    else:
        raise SystemExit("Base configuration file not specified! Exiting.")

    main(cfg)
