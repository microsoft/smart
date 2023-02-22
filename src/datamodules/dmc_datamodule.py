# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
from typing import Optional

import dmc2gym
import numpy as np
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from datamodules.dmc_replay_buffer import FullReplayBuffer
from models.utils import get_min_action_dmc


def create_selected_dataset(
    domain_name, task_name, data_dir_prefix, num_steps, replay_id=1, stack_size=4, select_rate=0.1, rand_select=False
):
    obss, actions, returns, done_idxs, rtgs, timesteps, step_returns = create_dataset(
        domain_name, task_name, data_dir_prefix, num_steps, replay_id, stack_size
    )
    step_returns[np.where(step_returns > 0)] = 1
    step_returns[np.where(step_returns < 0)] = 0
    print("step reward max", np.max(step_returns))
    print("step reward min", np.min(step_returns))

    returns = returns[:-1]  # cut off the 0 in the end
    sorting_index = np.argsort(returns)[::-1]  # sort from high to low
    n_select = max(int(select_rate * len(returns)), 1)
    print("selecting", n_select, "trajs from ", len(returns), "trajs")
    if not rand_select:
        selected_index = sorting_index[:n_select]
        print("selected index", selected_index)
    else:
        selected_index = np.random.choice(sorting_index, n_select, replace=False)
        print("selected random index", selected_index)

    selected_timesteps = []
    selected_obss = None
    selected_actions = None
    selected_returns = []
    selected_rtgs = []
    selected_step_returns = []
    selected_dones = []

    for k, idx in enumerate(selected_index):
        left = done_idxs[idx - 1] if idx > 0 else 0
        right = done_idxs[idx]
        if left == 0:
            selected_timesteps = np.concatenate((selected_timesteps, timesteps[left:right]))
        else:
            selected_timesteps = np.concatenate((selected_timesteps, timesteps[left + 1 : right + 1]))
        # print("left", left, "right", right, selected_timesteps)
        if selected_obss is None:
            selected_obss = obss[left:right]
            selected_actions = actions[left:right]
        else:
            selected_obss = np.concatenate((selected_obss, obss[left:right]), axis=0)
            selected_actions = np.concatenate((selected_actions, actions[left:right]), axis=0)
        selected_rtgs = np.concatenate((selected_rtgs, rtgs[left:right]))
        selected_step_returns = np.concatenate((selected_step_returns, step_returns[left:right]))
        selected_returns.append(returns[idx])
        selected_dones.append(len(selected_obss))

    return (
        selected_obss,
        selected_actions,
        selected_returns,
        selected_dones,
        selected_rtgs,
        selected_timesteps,
        selected_step_returns,
    )


def create_dataset(domain_name, task_name, data_dir_prefix, num_steps, replay_id=1, stack_size=4):
    env = dmc2gym.make(
        domain_name=domain_name,
        task_name=task_name,
        visualize_reward=False,
        from_pixels=False,
        height=84,
        width=84,
        frame_skip=4,
    )

    replay_buffer = FullReplayBuffer(
        obs_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        state_shape=env.get_physics_shape(),
        pixel_shape=env.get_pixel_space().shape,
        capacity=num_steps,
        batch_size=128,
        device="cpu",
    )

    exp_name = domain_name + "_" + task_name + "_s" + str(replay_id)
    load_path = os.path.join(data_dir_prefix, exp_name + "/" + exp_name + "/buffer")
    print("loading data from path", load_path)

    replay_buffer.load(load_path)
    idx = min(replay_buffer.idx, num_steps)
    print("loaded", idx, "transitions")

    obss = replay_buffer.pixels[:idx]
    actions = replay_buffer.actions[:idx]
    stepwise_returns = replay_buffer.rewards[:idx].flatten()
    returns = []
    done_idxs = []
    timesteps = np.zeros(len(actions), dtype=int)
    start = 0
    for i in range(1, idx + 1):
        if not replay_buffer.not_dones[i - 1] or i % 250 == 0:
            done_idxs.append(i)
            returns.append(np.sum(stepwise_returns[start:i]))
            timesteps[start:i] = np.arange(i - start)
            start = i
    print("max timestep is %d" % max(timesteps))

    # for tmp in range(idx):
    #     obs = obss[tmp]
    #     obs = obs / 255.
    #     obs = np.transpose(obs, [1, 2, 0])
    #     plt.imsave(os.path.join("/data2/plots/{}.png".format(tmp)), obs)

    returns = np.array(returns)
    done_idxs = np.array(done_idxs)

    # -- create reward-to-go dataset
    start_index = 0
    rtg = np.zeros_like(stepwise_returns)
    for i in done_idxs:
        i = int(i)
        curr_traj_returns = stepwise_returns[start_index:i]
        for j in range(i - 1, start_index - 1, -1):  # start from i-1
            rtg_j = curr_traj_returns[j - start_index : i - start_index]
            rtg[j] = sum(rtg_j)
        start_index = i
    print("max rtg is %d" % max(rtg))

    # # -- create timestep dataset
    # start_index = 0
    # timesteps = np.zeros(len(actions)+1, dtype=int)
    # for i in done_idxs:
    #     i = int(i)
    #     timesteps[start_index:i+1] = np.arange(i+1 - start_index)
    #     start_index = i+1

    # print("obs", obss.shape)
    # print("actions", actions)
    # print("returns", returns.shape)
    # print("done_idxs", done_idxs.shape)
    # print("rtg", rtg.shape)
    # print("ts", timesteps.shape)
    # print("sw return", stepwise_returns.shape)

    return obss, actions, returns, done_idxs, rtg, timesteps, stepwise_returns


class StateActionReturnDataset(Dataset):
    def __init__(self, data, context_length, actions, done_idxs, rtgs, timesteps, step_returns):
        self.context_length = context_length
        self.vocab_size = len(actions[0])
        self.data = data
        self.actions = actions
        self.done_idxs = done_idxs
        self.rtgs = rtgs
        self.timesteps = timesteps
        # self.next_obss = next_obss
        self.step_returns = step_returns

    def __len__(self):
        return len(self.data) - self.context_length * 2

    def __getitem__(self, idx):
        done_idx = idx + self.context_length
        for i in self.done_idxs:
            if i > idx:  # first done_idx greater than idx
                done_idx = min(int(i), done_idx)
                break
        idx = done_idx - self.context_length
        states = torch.tensor(np.array(self.data[idx:done_idx]), dtype=torch.float32).reshape(
            self.context_length, -1
        )  # (block_size, 4*84*84)
        states = states / 255.0
        actions = torch.tensor(self.actions[idx:done_idx], dtype=torch.float32)  # (block_size, 1)
        rtgs = torch.tensor(self.rtgs[idx:done_idx], dtype=torch.float32).unsqueeze(1)
        timesteps = torch.tensor(self.timesteps[idx : idx + 1], dtype=torch.int64).unsqueeze(1)

        # next_states = torch.tensor(np.array(self.next_obss[idx:done_idx]), dtype=torch.float32).reshape(block_size, -1) # (block_size, 4*84*84)
        # next_states = next_states / 255.
        # next_actions = torch.tensor(self.next_actions[idx:done_idx], dtype=torch.long).unsqueeze(1) # (block_size, 1)
        step_returns = torch.tensor(self.step_returns[idx:done_idx], dtype=torch.float32).unsqueeze(1)

        return states, actions, rtgs, timesteps, step_returns


class MultiTaskStateActionReturnDataset(Dataset):
    def __init__(self, data, context_length, actions, done_idxs, rtgs, timesteps, step_returns, task_splits):
        self.context_length = context_length
        self.vocab_size = len(actions[0])
        self.data = data
        self.actions = actions
        self.done_idxs = done_idxs
        self.rtgs = rtgs
        self.timesteps = timesteps
        # self.next_obss = next_obss
        self.step_returns = step_returns
        self.task_splits = task_splits  # len(task_splits): num_tasks, task_splits[i]: the i-th task end index

    def __len__(self):
        return len(self.data) - self.context_length * 2 * len(self.task_splits)

    def __getitem__(self, idx):
        done_idx = idx + self.context_length
        for i in self.done_idxs:
            if i > idx:  # first done_idx greater than idx
                done_idx = min(int(i), done_idx)
                break
        for task_id, j in enumerate(self.task_splits):
            if j > idx:
                done_idx = min(int(j), done_idx)
                break
        idx = done_idx - self.context_length
        states = torch.tensor(np.array(self.data[idx:done_idx]), dtype=torch.float32).reshape(
            self.context_length, -1
        )  # (block_size, 3*84*84)
        states = states / 255.0
        actions = torch.tensor(self.actions[idx:done_idx], dtype=torch.float32)  # (block_size, 1)
        rtgs = torch.tensor(self.rtgs[idx:done_idx], dtype=torch.float32).unsqueeze(1)
        timesteps = torch.tensor(self.timesteps[idx : idx + 1], dtype=torch.int64).unsqueeze(1)

        # next_states = torch.tensor(np.array(self.next_obss[idx:done_idx]), dtype=torch.float32).reshape(block_size, -1) # (block_size, 4*84*84)
        # next_states = next_states / 255.
        # next_actions = torch.tensor(self.next_actions[idx:done_idx], dtype=torch.long).unsqueeze(1) # (block_size, 1)
        step_returns = torch.tensor(self.step_returns[idx:done_idx], dtype=torch.float32).unsqueeze(1)

        return states, actions, rtgs, timesteps, step_returns, task_id


class DMCDataModule(LightningDataModule):
    """Example of LightningDataModule for Atari DQN Replay Buffer dataset.

    A DataModule implements 5 key methods:
        - prepare_data (things to do on 1 GPU/TPU, not on every GPU/TPU in distributed mode)
        - setup (things to do on every accelerator in distributed mode)
        - train_dataloader (the training dataloader)
        - val_dataloader (the validation dataloader(s))
        - test_dataloader (the test dataloader(s))

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html
    """

    def __init__(
        self,
        domain,
        task,
        data_dir_prefix,
        context_length=30,
        num_buffers=50,
        num_steps=500000,
        trajectories_per_buffer=10,
        stack_size=4,
        batch_size=128,
        num_workers=1,
        train_replay_id=1,
        val_replay_id=2,
        select_rate=0,  # not used. only to match signature with DMCBCDataModule
        rand_select=False,  # not used. only to match signature with DMCBCDataModule
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        self.save_hyperparameters(logger=False)
        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None

        if self.hparams.domain not in self.hparams.data_dir_prefix:
            self.hparams.data_dir_prefix = os.path.join(
                self.hparams.data_dir_prefix, "randcollect_" + self.hparams.domain + "_" + self.hparams.task
            )

    @property
    def context_length(self) -> int:
        return self.context_length

    def prepare_data(self):
        """Download data if needed.

        This method is called only from a single GPU.
        Do not use it to assign state (self.x = y).
        """
        # TODO: check whether data exists in the given dir
        train_name = self.hparams.domain + "_" + self.hparams.task + "_s" + str(self.hparams.train_replay_id)
        val_name = self.hparams.domain + "_" + self.hparams.task + "_s" + str(self.hparams.val_replay_id)
        train_dir = os.path.join(self.hparams.data_dir_prefix, f"{train_name}/{train_name}/buffer")
        val_dir = os.path.join(self.hparams.data_dir_prefix, f"{val_name}/{val_name}/buffer")

        assert os.path.exists(train_dir), f"Must download data to {train_dir}"
        assert os.path.exists(val_dir), f"Must download data to {val_dir}"

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning when doing `trainer.fit()` and `trainer.test()`,
        so be careful not to execute the random split twice! The `stage` can be used to
        differentiate whether it's called before trainer.fit()` or `trainer.test()`.
        """
        if not self.train_dataset:
            obss, actions, _, done_idxs, rtgs, timesteps, step_returns = create_dataset(
                domain_name=self.hparams.domain,
                task_name=self.hparams.task,
                data_dir_prefix=self.hparams.data_dir_prefix,
                num_steps=self.hparams.num_steps,
                replay_id=self.hparams.train_replay_id,
            )
            step_returns[np.where(step_returns > 0)] = 1
            step_returns[np.where(step_returns < 0)] = 0
            print("step reward max", np.max(step_returns))
            print("step reward min", np.min(step_returns))

            self.train_dataset = StateActionReturnDataset(
                obss, self.hparams.context_length, actions, done_idxs, rtgs, timesteps, step_returns
            )

        if not self.val_dataset:
            obss, actions, _, done_idxs, rtgs, timesteps, step_returns = create_dataset(
                domain_name=self.hparams.domain,
                task_name=self.hparams.task,
                data_dir_prefix=self.hparams.data_dir_prefix,
                num_steps=self.hparams.num_steps,
                replay_id=self.hparams.val_replay_id,
            )
            step_returns[np.where(step_returns > 0)] = 1
            step_returns[np.where(step_returns < 0)] = 0

            self.val_dataset = StateActionReturnDataset(
                obss, self.hparams.context_length, actions, done_idxs, rtgs, timesteps, step_returns
            )
            # self.test_dataset = StateActionReturnDataset(
            #     obss[:self.hparams.batch_size], self.hparams.context_length, actions[:self.hparams.batch_size], done_idxs[:self.hparams.batch_size], rtgs[:self.hparams.batch_size], timesteps[:self.hparams.batch_size], step_returns[:self.hparams.batch_size]
            # )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            pin_memory=True,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            shuffle=False,
            pin_memory=True,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            persistent_workers=True,
        )

    def test_dataloader(self):
        return None
        # return DataLoader(
        #     self.test_dataset,
        #     shuffle=False,
        #     pin_memory=True,
        #     batch_size=self.hparams.batch_size,
        #     num_workers=self.hparams.num_workers,
        #     persistent_workers=True
        # )


class DMCBCDataModule(LightningDataModule):
    """Example of LightningDataModule for behavior cloning on DMC. Only top 10 percent trajectories
    are used.

    A DataModule implements 5 key methods:
        - prepare_data (things to do on 1 GPU/TPU, not on every GPU/TPU in distributed mode)
        - setup (things to do on every accelerator in distributed mode)
        - train_dataloader (the training dataloader)
        - val_dataloader (the validation dataloader(s))
        - test_dataloader (the test dataloader(s))

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html
    """

    def __init__(
        self,
        domain,
        task,
        data_dir_prefix,
        context_length=30,
        num_buffers=50,
        num_steps=500000,
        trajectories_per_buffer=10,
        stack_size=4,
        batch_size=128,
        num_workers=1,
        train_replay_id=1,
        val_replay_id=2,
        select_rate=0.1,
        rand_select=False,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        self.save_hyperparameters(logger=False)
        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None

        if self.hparams.domain not in self.hparams.data_dir_prefix:
            self.hparams.data_dir_prefix = os.path.join(
                self.hparams.data_dir_prefix, "fullcollect_" + self.hparams.domain + "_" + self.hparams.task
            )

    @property
    def context_length(self) -> int:
        return self.context_length

    def prepare_data(self):
        """Download data if needed.

        This method is called only from a single GPU.
        Do not use it to assign state (self.x = y).
        """
        # TODO: check whether data exists in the given dir
        train_name = self.hparams.domain + "_" + self.hparams.task + "_s" + str(self.hparams.train_replay_id)
        val_name = self.hparams.domain + "_" + self.hparams.task + "_s" + str(self.hparams.val_replay_id)
        train_dir = os.path.join(self.hparams.data_dir_prefix, f"{train_name}/{train_name}/buffer")
        val_dir = os.path.join(self.hparams.data_dir_prefix, f"{val_name}/{val_name}/buffer")

        assert os.path.exists(train_dir), f"Must download data to {train_dir}"
        assert os.path.exists(val_dir), f"Must download data to {val_dir}"

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning when doing `trainer.fit()` and `trainer.test()`,
        so be careful not to execute the random split twice! The `stage` can be used to
        differentiate whether it's called before trainer.fit()` or `trainer.test()`.
        """
        if not self.train_dataset:
            obss, actions, returns, done_idxs, rtgs, timesteps, step_returns = create_selected_dataset(
                domain_name=self.hparams.domain,
                task_name=self.hparams.task,
                data_dir_prefix=self.hparams.data_dir_prefix,
                num_steps=self.hparams.num_steps,
                replay_id=self.hparams.train_replay_id,
                select_rate=self.hparams.select_rate,
                rand_select=self.hparams.rand_select,
            )

            print("selected returns", len(returns), returns)
            print("actions", len(actions))
            print("obss", len(obss))
            print("rtgs", len(rtgs))
            print("stepret", len(step_returns))
            print("dones", done_idxs)

            self.train_dataset = StateActionReturnDataset(
                obss, self.hparams.context_length, actions, done_idxs, rtgs, timesteps, step_returns
            )

        if not self.val_dataset:
            obss, actions, _, done_idxs, rtgs, timesteps, step_returns = create_selected_dataset(
                domain_name=self.hparams.domain,
                task_name=self.hparams.task,
                data_dir_prefix=self.hparams.data_dir_prefix,
                num_steps=self.hparams.num_steps,
                replay_id=self.hparams.val_replay_id,
                select_rate=self.hparams.select_rate,
            )
            step_returns[np.where(step_returns > 0)] = 1
            step_returns[np.where(step_returns < 0)] = 0

            self.val_dataset = StateActionReturnDataset(
                obss, self.hparams.context_length, actions, done_idxs, rtgs, timesteps, step_returns
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            pin_memory=True,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            shuffle=False,
            pin_memory=True,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            persistent_workers=True,
        )

    def test_dataloader(self):
        return None


class DMCMultiDomainDataModule(LightningDataModule):
    """Example of LightningDataModule for pretraining in multiple domains and multiple tasks on
    DMC."""

    def __init__(
        self,
        source_envs,
        data_dir_prefix,
        context_length=30,
        num_buffers=50,
        num_steps=500000,
        trajectories_per_buffer=10,
        stack_size=4,
        batch_size=128,
        num_workers=1,
        train_replay_id=1,
        val_replay_id=2,
        select_rate=0.1,
        seed=123,
        source_data_type=["full"],
        biased_multi=False,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        self.save_hyperparameters(logger=False)

        self.domains = source_envs.keys()
        self.action_dims = [get_min_action_dmc(domain) for domain in self.domains]
        print("domains", self.domains)
        print("action dims", self.action_dims)

        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None

        if self.hparams.source_data_type == "full":
            self.hparams.dataset_types = ["fullcollect"]
        elif self.hparams.source_data_type == "rand":
            self.hparams.dataset_types = ["randcollect"]
        elif self.hparams.source_data_type == "mix":
            self.hparams.dataset_types = ["fullcollect", "randcollect"]
            self.hparams.num_steps = self.hparams.num_steps // 2

    @property
    def context_length(self) -> int:
        return self.context_length

    def prepare_data(self):
        """Download data if needed.

        This method is called only from a single GPU.
        Do not use it to assign state (self.x = y).
        """
        # check whether data exists in the given dir
        for domain in self.domains:
            for task in self.hparams.source_envs[domain]:
                for dataset_type in self.hparams.dataset_types:
                    data_dir_prefix = os.path.join(self.hparams.data_dir_prefix, f"{dataset_type}_{domain}_{task}")
                    for agent_id in range(1, self.hparams.train_replay_id + 1):
                        train_name = domain + "_" + task + "_s" + str(agent_id)
                        train_dir = os.path.join(data_dir_prefix, f"{train_name}/{train_name}/buffer")

                        assert os.path.exists(train_dir), f"Must download data to {train_dir}"

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning when doing `trainer.fit()` and `trainer.test()`,
        so be careful not to execute the random split twice! The `stage` can be used to
        differentiate whether it's called before trainer.fit()` or `trainer.test()`.
        """
        if not self.train_dataset:
            obss, actions, done_idxs, rtgs, timesteps, step_returns = [], [], [], [], [], []
            sum_samples = 0
            task_splits = []

            for domain_idx, domain in enumerate(self.domains):
                tasks = self.hparams.source_envs[domain]
                n_step = self.hparams.num_steps
                if self.hparams.biased_multi:
                    if domain == "cartpole":
                        n_step = self.hparams.num_steps // 2
                    elif domain == "walker":
                        n_step = self.hparams.num_steps + 10000
                print("loading", n_step, "per task-replay from", domain)
                for task in tasks:
                    task_splits.append(sum_samples)
                    for dataset_type in self.hparams.dataset_types:
                        for i in range(self.hparams.train_replay_id):
                            o, a, _, d, rtg, t, sr = create_dataset(
                                domain_name=domain,
                                task_name=task,
                                data_dir_prefix=os.path.join(
                                    self.hparams.data_dir_prefix, f"{dataset_type}_{domain}_{task}"
                                ),
                                num_steps=n_step,
                                replay_id=i + 1,
                            )
                            obss.append(o)

                            left_pad = sum(self.action_dims[:domain_idx])
                            right_pad = sum(self.action_dims[domain_idx + 1 :])
                            a = np.pad(a, ((0, 0), (left_pad, right_pad)), "constant")

                            actions.append(a)
                            done_idxs.append(d)
                            rtgs.append(rtg)
                            timesteps.append(t)
                            sr[np.where(sr > 0)] = 1
                            sr[np.where(sr < 0)] = 0
                            step_returns.append(sr)

                            task_splits[-1] += len(o)
                            sum_samples += len(o)

            obss = np.concatenate([obs for obs in obss], axis=0)
            actions = np.concatenate([action for action in actions], axis=0)
            done_idxs = np.concatenate([done_idx for done_idx in done_idxs], axis=0)
            rtgs = np.concatenate([rtg for rtg in rtgs], axis=0)
            timesteps = np.concatenate([timestep for timestep in timesteps], axis=0)
            step_returns = np.concatenate([step_return for step_return in step_returns], axis=0)

            # task_splits = np.arange(1, len(self.hparams.tasks)+1, dtype=np.int32) * self.hparams.num_steps * self.hparams.train_replay_id
            print("task splits", task_splits)
            print("total number of transitions", len(obss))
            print("action shape", actions.shape)
            # assert task_splits[-1] == len(obss), "number of loaded trasitions does not match: {} != {}".format(task_splits[-1], len(obss))

            self.train_dataset = MultiTaskStateActionReturnDataset(
                obss, self.hparams.context_length, actions, done_idxs, rtgs, timesteps, step_returns, task_splits
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            pin_memory=True,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.train_dataset,
            shuffle=False,
            pin_memory=True,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            persistent_workers=True,
        )

    def test_dataloader(self):
        return None
