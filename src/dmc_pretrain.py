# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os

from omegaconf import OmegaConf

from datamodules.dummy_datamodule import RandomDataset
from models.multitask_ct_module import MultiTaskCTLitModule
from utils import pl_utils


def main(cfg):
    trainer = pl_utils.instantiate_trainer(cfg)
    datamodule = pl_utils.instantiate_class(cfg["data"])

    if cfg["load_model_from"] is not None:
        model = MultiTaskCTLitModule.load_from_checkpoint(cfg.load_model_from, **cfg.model)
        print("loaded model from", cfg["load_model_from"])
    else:
        model = pl_utils.instantiate_class(cfg["model"])

    # init root dir
    os.makedirs(cfg["output_dir"], exist_ok=True)
    print("output dir", cfg["output_dir"])

    # start training
    trainer.fit(model, datamodule)

    # testing
    dummy = RandomDataset()
    trainer.test(model, dummy)


if __name__ == "__main__":
    cfg = OmegaConf.from_cli()

    if "base" in cfg:
        basecfg = OmegaConf.load(cfg.base)
        del cfg.base
        cfg = OmegaConf.merge(basecfg, cfg)
        cfg = OmegaConf.to_container(cfg, resolve=True)
        print(OmegaConf.to_yaml(cfg))
        main(cfg)
    else:
        raise SystemExit("Base configuration file not specified! Exiting.")

    main(cfg)
