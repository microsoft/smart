# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from collections import OrderedDict
from typing import Any, Dict, List

import torch
from pytorch_lightning import LightningModule

from models.components.mingpt import GPT, GPTConfig
from models.utils import get_min_action_dmc


class MultiTaskCTLitModule(LightningModule):
    """LightningModule for multi-task control transformer."""

    def __init__(
        self,
        epochs: int,
        agent_type: str,
        model_type: str,
        timestep: int,
        n_embd: int,
        lr: float,
        forward: bool,
        inverse: bool,
        reward: bool,
        rand_inverse: bool,
        unsupervise: bool,
        rand_mask_size: int,
        freeze_encoder: bool,
        context_length: int,
        betas: List[float],
        weight_decay: float,
        n_layer: int,
        n_head: int,
        mask_obs_size: int,
        pred_layers: int,
        bc_layers: int,
        rtg_layers: int,
        forward_weight: float,
        source_envs: Dict[str, List[str]],
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # it also ensures init params will be stored in ckpt
        self.save_hyperparameters()
        channels = 3
        if hasattr(self.hparams, "source_envs"):
            self.domains = self.hparams.source_envs.keys()
            vocab_size = 0
            ## concatenate the action spaces
            for domain in self.domains:
                vocab_size += get_min_action_dmc(domain)
        else:
            vocab_size = get_min_action_dmc(self.hparams.domain)

        if self.hparams.agent_type == "gpt":
            block_size = self.hparams.context_length * 2
            mconf = GPTConfig(
                vocab_size,
                block_size,
                max_timestep=self.hparams.timestep,
                channels=channels,
                model_type=self.hparams.model_type,
                n_layer=self.hparams.n_layer,
                n_head=self.hparams.n_head,
                n_embd=self.hparams.n_embd,
                cont_action=True,
                pred_layers=self.hparams.pred_layers,
                rtg_layers=self.hparams.rtg_layers,
                bc_layers=self.hparams.bc_layers,
            )
            self.net = GPT(mconf)
            print(self.net)
            print(self.hparams)
        else:
            assert "agent type not supported"

    def training_step(self, batch: Any, batch_idx: int):
        obs, actions, rtg, ts, rewards, task_ids = batch
        targets = None if self.hparams.unsupervise else actions

        if self.hparams.rand_mask_size < 0:
            rand_mask_size = max(1, int(self.hparams.context_length * (self.current_epoch + 1) / self.hparams.epochs))
            self.log(f"train/mask_size", rand_mask_size, on_step=False, on_epoch=True, prog_bar=False)
        else:
            rand_mask_size = self.hparams.rand_mask_size

        if self.hparams.mask_obs_size < 0:
            mask_obs_size = min(
                self.hparams.context_length // 2,
                max(1, int(self.hparams.context_length * (self.current_epoch + 1) / self.hparams.epochs)),
            )
            self.log(f"train/mask_obs_size", mask_obs_size, on_step=False, on_epoch=True, prog_bar=False)
        else:
            mask_obs_size = self.hparams.mask_obs_size

        logits, all_losses = self.net(
            obs,
            actions,
            targets,
            rtg,
            ts,
            rewards,
            pred_forward=self.hparams.forward,
            pred_inverse=self.hparams.inverse,
            pred_reward=self.hparams.reward,
            pred_rand_inverse=self.hparams.rand_inverse,
            rand_mask_size=rand_mask_size,
            mask_obs_size=mask_obs_size,
            forward_weight=self.hparams.forward_weight,
        )

        avg_loss = 0
        for name, loss in all_losses.items():
            self.log(f"train/{name}", loss, on_step=True, on_epoch=True, prog_bar=False)
            avg_loss += loss
        avg_loss /= len(all_losses.keys())
        # log train metrics
        self.log("train/avg_loss", avg_loss, on_step=False, on_epoch=True, prog_bar=False)

        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()` below
        # remember to always return loss from `training_step()` or else backpropagation will fail!
        return {"loss": avg_loss, "logits": logits}

    def validation_step(self, batch: Any, batch_idx: int):
        obs, actions, rtg, ts, rewards, task_ids = batch
        targets = None if self.hparams.unsupervise else actions

        if self.hparams.rand_mask_size < 0:
            rand_mask_size = max(1, int(self.hparams.context_length * (self.current_epoch + 1) / self.hparams.epochs))
            self.log(f"train/mask_size", rand_mask_size, on_step=False, on_epoch=True, prog_bar=False)
        else:
            rand_mask_size = self.hparams.rand_mask_size

        if self.hparams.mask_obs_size < 0:
            mask_obs_size = min(
                self.hparams.context_length // 2,
                max(1, int(self.hparams.context_length * (self.current_epoch + 1) / self.hparams.epochs)),
            )
            self.log(f"train/mask_obs_size", mask_obs_size, on_step=False, on_epoch=True, prog_bar=False)
        else:
            mask_obs_size = self.hparams.mask_obs_size

        logits, all_losses = self.net(
            obs,
            actions,
            targets,
            rtg,
            ts,
            rewards,
            pred_forward=self.hparams.forward,
            pred_inverse=self.hparams.inverse,
            pred_reward=self.hparams.reward,
            pred_rand_inverse=self.hparams.rand_inverse,
            rand_mask_size=rand_mask_size,
            mask_obs_size=mask_obs_size,
        )

        avg_loss = 0
        for name, loss in all_losses.items():
            self.log(f"val/{name}", loss, on_step=True, on_epoch=True, prog_bar=False)
            avg_loss += loss

        avg_loss /= len(all_losses.keys())

        # log train metrics
        self.log("val/avg_loss", avg_loss, on_step=False, on_epoch=True, prog_bar=False)

        return {"loss": avg_loss, "logits": logits}

    def test_step(self, batch: Any, batch_idx: int):
        pass

    def test_epoch_end(self, outputs: List[Any]):
        pass

    def configure_optimizers(self):
        if self.hparams.freeze_encoder:
            return self.net.configure_naive_optimizer(self.hparams)
        else:
            return self.net.configure_optimizers(self.hparams)

    def load_state_dict(self, state_dict: "OrderedDict[str, torch.Tensor]", strict: bool = True):
        if self.hparams.agent_type == "initgpt":
            state_dict.pop("net.pos_emb")
            state_dict.pop("net.mask")
            state_dict.pop("net.inverse_mask")
            strict = False
        return super().load_state_dict(state_dict, strict)
