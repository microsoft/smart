# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""The MIT License (MIT) Copyright (c) 2020 Andrej Karpathy.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import random

import numpy as np
import torch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


def top_k_logits(logits, k):
    v, ix = torch.topk(logits, k)
    out = logits.clone()
    out[out < v[:, [-1]]] = -float("Inf")
    return out


def get_min_action_dmc(domain):
    domain_to_actions = {
        "cheetah": 6,
        "walker": 6,
        "hopper": 4,
        "cartpole": 1,
        "acrobot": 1,
        "pendulum": 1,
        "finger": 2,
    }
    if domain in domain_to_actions.keys():
        return domain_to_actions[domain]
    else:
        raise NotImplementedError()


def get_exp_return_dmc(domain, task):
    game_to_returns = {
        "cheetah_run": 850,
        "walker_stand": 980,
        "walker_walk": 950,
        "walker_run": 700,
        "hopper_stand": 900,
        "hopper_hop": 200,
        "cartpole_swingup": 875,
        "cartpole_balance": 1000,
        "pendulum_swingup": 1000,
        "finger_turn_easy": 1000,
        "finger_turn_hard": 1000,
        "finger_spin": 800,
    }
    if domain + "_" + task in game_to_returns.keys():
        return game_to_returns[domain + "_" + task]
    else:
        raise NotImplementedError()
