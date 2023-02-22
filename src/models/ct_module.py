# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from collections import OrderedDict
from typing import Any, List

import numpy as np
import torch
from pytorch_lightning import LightningModule
from torch.nn import functional as F

from models.components.mingpt import GPT, GPTConfig
from models.utils import get_exp_return_dmc, get_min_action_dmc, top_k_logits


class CTLitModule(LightningModule):
    """LightningModule for Control Transformer."""

    def __init__(
        self,
        agent_type: str,
        model_type: str,
        domain: str,
        task: str,
        n_embd: int,
        lr: float,
        unsupervise: bool,
        forward: bool,
        inverse: bool,
        reward: bool,
        rand_inverse: bool,
        freeze_encoder: bool,
        rand_attn_only: bool,
        rand_mask_size: bool,
        mask_obs_size: bool,
        forward_weight: float,
        n_layer: int,
        n_head: int,
        rtg_layers: int,
        bc_layers: int,
        pred_layers: int,
        context_length: int,
        epochs: int,
        timestep: int,
        weight_decay: float,
        betas: List[float],
        eval_epochs: int,
        seed: int,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # it also ensures init params will be stored in ckpt
        self.save_hyperparameters()
        channels = 3
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
        else:
            assert "agent type not supported"

    def load_my_checkpoint(self, path, no_action=False, strict=True, no_action_head=False):
        m = torch.load(path)["state_dict"]
        model_dict = self.state_dict()
        for k in m.keys():
            if no_action:
                if (
                    "reward_conditioned_head" in k
                    or "naive_head" in k
                    or "inverse_pred_head" in k
                    or "rand_inverse_pred_head" in k
                    or "action_encoder" in k
                    or "tok_emb" in k
                ):
                    continue
            if no_action_head:
                if "reward_conditioned_head" in k:
                    continue

            if k in model_dict:
                pname = k
                pval = m[k]
                model_dict[pname] = pval.clone().to(model_dict[pname].device)

        self.load_state_dict(model_dict, strict=strict)

    def training_step(self, batch: Any, batch_idx: int):
        obs, actions, rtg, ts, rewards = batch
        targets = None if self.hparams.unsupervise else actions

        if self.hparams.rand_mask_size < 0:
            rand_mask_size = max(1, int(self.hparams.context_length * (self.current_epoch + 1) / self.hparams.epochs))
            self.log(f"train/mask_size", rand_mask_size, on_step=False, on_epoch=True, prog_bar=False)
        else:
            rand_mask_size = self.hparams.rand_mask_size

        if self.hparams.mask_obs_size < 0:
            mask_obs_size = max(
                1, int(self.hparams.context_length * 0.5 * (self.current_epoch + 1) / self.hparams.epochs)
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
        obs, actions, rtg, ts, rewards = batch
        targets = None if self.hparams.unsupervise else actions

        if self.hparams.rand_mask_size < 0:
            rand_mask_size = max(1, int(self.hparams.context_length * (self.current_epoch + 1) / self.hparams.epochs))
            self.log(f"train/mask_size", rand_mask_size, on_step=False, on_epoch=True, prog_bar=False)
        else:
            rand_mask_size = self.hparams.rand_mask_size

        if self.hparams.mask_obs_size < 0:
            mask_obs_size = max(
                1, int(self.hparams.context_length * 0.5 * (self.current_epoch + 1) / self.hparams.epochs)
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
            self.log(f"val/{name}", loss, on_step=False, on_epoch=True, prog_bar=False)
            avg_loss += loss

        avg_loss /= len(all_losses.keys())

        # log train metrics
        self.log("val/avg_loss", avg_loss, on_step=False, on_epoch=True, prog_bar=False)

        return {"loss": avg_loss, "logits": logits}

    def validation_epoch_end(self, outputs: List[Any]):
        eval_return, _ = self.get_return_dmc(self.hparams.eval_epochs)
        self.log("val/interactive_reward", eval_return, on_step=False, on_epoch=True, prog_bar=False)

    def test_step(self, batch: Any, batch_idx: int):
        eval_return, std_return = self.get_return_dmc(self.hparams.eval_epochs)
        self.log("test/interactive_reward", eval_return, on_step=False, on_epoch=True)
        self.log("test/std", std_return, on_step=False, on_epoch=True)
        return {"loss": eval_return}

    def get_return_dmc(self, epochs):
        import dmc2gym

        env = dmc2gym.make(
            domain_name=self.hparams.domain,
            task_name=self.hparams.task,
            visualize_reward=False,
            from_pixels=True,
            height=84,
            width=84,
            frame_skip=4,
            version=2,
        )
        env.seed(self.hparams.seed)

        if self.hparams.model_type == "reward_conditioned":
            ret = get_exp_return_dmc(self.hparams.domain, self.hparams.task)
        else:
            ret = 0

        T_rewards = []
        done = True
        for i in range(epochs):
            state = env.reset()
            state = torch.from_numpy(state).type(torch.float32).to(self.device).div_(255).unsqueeze(0).unsqueeze(0)
            # print("test obs", state.size())
            # print(state)
            # print(obs)
            rtgs = [ret]
            # first state is from env, first rtg is target return, and first timestep is 0
            sampled_action = self.sample(
                state,
                1,
                actions=None,
                rtgs=torch.tensor(rtgs, dtype=torch.float32).to(self.device).unsqueeze(0).unsqueeze(-1),
                timesteps=torch.zeros((1, 1, 1), dtype=torch.int64).to(self.device),
            )

            j = 0
            all_states = state
            actions = []
            while True:
                if done:
                    state, reward_sum, done = env.reset(), 0, False
                action = sampled_action.cpu().numpy()[0]
                action = np.clip(action, -1, 1).astype(np.float32)
                actions += [sampled_action]
                state, reward, done, _ = env.step(action)
                reward_sum += reward
                j += 1

                if done:
                    T_rewards.append(reward_sum)
                    break

                state = torch.from_numpy(state).type(torch.float32).to(self.device).div_(255).unsqueeze(0).unsqueeze(0)

                all_states = torch.cat([all_states, state], dim=1)

                rtgs += [rtgs[-1] - reward]
                # all_states has all previous states and rtgs has all previous rtgs (will be cut to block_size in utils.sample)
                # timestep is just current timestep
                past_actions = torch.cat(actions, dim=0)
                sampled_action = self.sample(
                    all_states,
                    1,
                    actions=past_actions.to(self.device).unsqueeze(0),
                    rtgs=torch.tensor(rtgs, dtype=torch.float32).to(self.device).unsqueeze(0).unsqueeze(-1),
                    timesteps=(
                        min(j, self.hparams.timestep) * torch.ones((1, 1, 1), dtype=torch.int64).to(self.device)
                    ),
                )
            print("episode", i, action, reward_sum)
        env.close()
        T_rewards = np.array(T_rewards)
        eval_return = T_rewards.mean()
        std_return = T_rewards.std()
        print("target return: %d, eval return: %.1f +- %.1f" % (ret, eval_return, std_return))

        return eval_return, std_return

    @torch.no_grad()
    def sample(
        self,
        x,
        steps,
        cont_action=True,
        temperature=1.0,
        sample=False,
        top_k=None,
        actions=None,
        rtgs=None,
        timesteps=None,
    ):
        """take a conditioning sequence of indices in x (of shape (b,t)) and predict the next token
        in the sequence, feeding the predictions back into the model each time.

        Clearly the sampling has quadratic complexity unlike an RNN that is only linear, and
        has a finite context window of block_size, unlike an RNN that has an infinite
        context window.
        """
        cont_length = self.hparams.context_length
        for k in range(steps):
            x_cond = x if x.size(1) <= cont_length else x[:, -cont_length:]  # crop context if needed
            if actions is not None:
                actions = (
                    actions if actions.size(1) <= cont_length else actions[:, -cont_length:]
                )  # crop context if needed
            rtgs = rtgs if rtgs.size(1) <= cont_length else rtgs[:, -cont_length:]  # crop context if needed
            logits, _ = self.net(x_cond, actions=actions, targets=None, rtgs=rtgs, timesteps=timesteps)
            if cont_action:
                x = logits[:, -1, :]
            else:
                # pluck the logits at the final step and scale by temperature
                logits = logits[:, -1, :] / temperature
                # optionally crop probabilities to only the top k options
                if top_k is not None:
                    logits = top_k_logits(logits, top_k)
                # apply softmax to convert to probabilities
                probs = F.softmax(logits, dim=-1)
                # sample from the distribution or take the most likely
                if sample:
                    ix = torch.multinomial(probs, num_samples=1)
                else:
                    _, ix = torch.topk(probs, k=1, dim=-1)
                # append to the sequence and continue
                # x = torch.cat((x, ix), dim=1)
                x = ix

        return x

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
