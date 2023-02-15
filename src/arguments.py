import argparse

parser = argparse.ArgumentParser()

## program and path
parser.add_argument("--data_dir_prefix", type=str, default="./data/")
parser.add_argument("--output_dir", type=str, default="./outputs/")
parser.add_argument("--exp_name", type=str, default="")
parser.add_argument("--load_model_from", type=str, default=None)
parser.add_argument("--no_load_action", default=False, action="store_true")
parser.add_argument("--no_strict", default=False, action="store_true")
parser.add_argument("--no_action_head", default=False, action="store_true")

## trainer (for pytorch_lightning)
parser.add_argument("--seed", type=int, default=123)
parser.add_argument("--accelerator", type=str, default="gpu")
parser.add_argument("--devices", type=int, default=1, help="how many GPUs to use")
parser.add_argument("--nodes", type=int, default=1)
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--save_k", type=int, default=5, help="how many checkpoints to save in finetuning")

## evaluate
parser.add_argument("--eval_epochs", type=int, default=50)

# dmc
parser.add_argument("--domain", type=str, default="cheetah")
parser.add_argument("--task", type=str, default="run")
parser.add_argument("--multi_config", type=str, default=None)

## data
parser.add_argument("--source_data_type", type=str, default="full", choices=["full", "rand", "mix"])
parser.add_argument("--context_length", type=int, default=30)
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--num_steps", type=int, default=500000)
parser.add_argument("--select_rate", type=float, default=0.1)
parser.add_argument("--train_replay_id", type=int, default=2)
parser.add_argument("--val_replay_id", type=int, default=5)
parser.add_argument("--timestep", type=int, default=10000)
parser.add_argument("--rand_select", default=False, action="store_true")
parser.add_argument("--biased_multi", default=False, action="store_true")



