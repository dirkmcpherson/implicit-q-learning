from utils import ReplayBuffer, eval_policy
import argparse
import IQL
import numpy as np
import torch
import os
import panda_gym
import gym

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Experiment
    parser.add_argument("--policy", default="IQL")  # Policy name
    parser.add_argument("--env", default="PandaPushModified-v2")  # OpenAI gym environment name
    parser.add_argument("--load_model_path", default="./models/last")  # OpenAI gym environment name
    parser.add_argument("--seed", default=1, type=int)  # Sets Gym, PyTorch and Numpy seeds
    # IQL
    parser.add_argument("--discount", default=0.99)  # Discount factor
    parser.add_argument("--tau", default=0.005)  # Target network update rate
    parser.add_argument("--expectile", default=0.8)  # Expectile parameter Tau
    parser.add_argument("--beta", default=3.0)  # Temperature parameter Beta
    parser.add_argument("--max_weight", default=100.0)  # Max weight for actor update
    args = parser.parse_args()


    [os.path.makedirs(path) for path in ["./results", "./models"] if not os.path.exists(path)]

    env = gym.make(args.env)
    # Set seeds
    env.seed(args.seed)
    env.action_space.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # state_dim = env.observation_space.shape[0]
    state_dim = env.observation_space["observation"].shape[0] + env.observation_space["desired_goal"].shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    hidden = (256, 256)

    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "discount": args.discount,
        "tau": args.tau,
        "expectile": args.expectile,
        "beta": args.beta,
        "max_weight": args.max_weight,
        "actor_hidden": hidden,
        "critic_hidden": hidden,
        "value_hidden": hidden,
        "deterministic_policy": False,
    }

    # Initialize policy
    policy = IQL.IQL(**kwargs)
    if args.load_model_path is not None:
        print(f"\tLoading policy from {args.load_model_path}")
        policy.load(args.load_model_path)

    offline_dataset_path = "./offline_buffers/PandaPushv2_buffer.npz"
    print(f"\tLoading offline data from {offline_dataset_path}")
    replay_buffer = ReplayBuffer(state_dim, action_dim)
    dataset = np.load(offline_dataset_path)
    replay_buffer.convert_npz(dataset)
    print(f"\t\tOffline dataset size: {len(replay_buffer)}")
    mean, std = replay_buffer.normalize_states()

    eval_policy(policy, args.env, args.seed, mean, std, render=True)