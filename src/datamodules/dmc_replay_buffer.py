# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os

import numpy as np
import torch


class FullReplayBuffer:
    """Buffer to store environment transitions."""

    def __init__(self, obs_shape, action_shape, state_shape, pixel_shape, capacity, batch_size, device):
        self.capacity = capacity
        self.batch_size = batch_size
        self.device = device

        # the proprioceptive obs is stored as float32, pixels obs as uint8
        obs_dtype = np.float32 if len(obs_shape) == 1 else np.uint8

        self.obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.next_obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones = np.empty((capacity, 1), dtype=np.float32)

        self.states = np.empty((capacity, *state_shape), dtype=np.float32)
        self.pixels = np.empty((capacity, *pixel_shape), dtype=np.uint8)

        self.idx = 0
        self.last_save = 0
        self.full = False

    def add(self, obs, action, reward, next_obs, done, state, pixel):
        np.copyto(self.obses[self.idx], obs)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.next_obses[self.idx], next_obs)
        np.copyto(self.not_dones[self.idx], not done)

        np.copyto(self.states[self.idx], state)
        np.copyto(self.pixels[self.idx], pixel)

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def sample(self):
        idxs = np.random.randint(0, self.capacity if self.full else self.idx, size=self.batch_size)

        obses = torch.as_tensor(self.obses[idxs], device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        next_obses = torch.as_tensor(self.next_obses[idxs], device=self.device).float()
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)

        return obses, actions, rewards, next_obses, not_dones

    def save(self, save_dir):
        if self.idx == self.last_save:
            return
        path = os.path.join(save_dir, "%d_%d.pt" % (self.last_save, self.idx))
        payload = [
            self.obses[self.last_save : self.idx],
            self.next_obses[self.last_save : self.idx],
            self.actions[self.last_save : self.idx],
            self.rewards[self.last_save : self.idx],
            self.not_dones[self.last_save : self.idx],
            self.states[self.last_save : self.idx],
            self.pixels[self.last_save : self.idx],
        ]
        self.last_save = self.idx
        torch.save(payload, path)

    def load(self, save_dir):
        chunks = os.listdir(save_dir)
        chucks = sorted(chunks, key=lambda x: int(x.split("_")[0]))
        for chunk in chucks:
            start, end = (int(x) for x in chunk.split(".")[0].split("_"))
            if end > self.capacity:
                break
            path = os.path.join(save_dir, chunk)
            payload = torch.load(path, map_location=self.device)
            assert self.idx == start
            self.obses[start:end] = payload[0]
            self.next_obses[start:end] = payload[1]
            self.actions[start:end] = payload[2]
            self.rewards[start:end] = payload[3]
            self.not_dones[start:end] = payload[4]
            self.states[start:end] = payload[5]
            self.pixels[start:end] = payload[6]
            self.idx = end

    def load_custom(self, save_dir, load_start, num_steps):
        chunks = os.listdir(save_dir)
        chucks = sorted(chunks, key=lambda x: int(x.split("_")[0]))
        for chunk in chucks:
            start, end = (int(x) for x in chunk.split(".")[0].split("_"))
            load_end = self.idx + end - start
            if start < load_start:
                continue
            if end > load_start + num_steps:
                break
            if load_end > self.capacity:
                print("Max capacity reached")
                break
            path = os.path.join(save_dir, chunk)
            payload = torch.load(path, map_location=self.device)
            self.obses[self.idx : load_end] = payload[0]
            self.next_obses[self.idx : load_end] = payload[1]
            self.actions[self.idx : load_end] = payload[2]
            self.rewards[self.idx : load_end] = payload[3]
            self.not_dones[self.idx : load_end] = payload[4]
            self.states[self.idx : load_end] = payload[5]
            self.pixels[self.idx : load_end] = payload[6]
            self.idx = load_end
            print("loaded till", self.idx)
