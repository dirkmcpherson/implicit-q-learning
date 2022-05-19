import numpy as np
import torch
import gym
import argparse
import os
import d4rl_pybullet
import h5py
from tqdm import tqdm
import time
import utils
import IQL
import panda_gym

# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, env_name, seed, mean, std, seed_offset=0, eval_episodes=10):
    eval_env = gym.make(env_name, render=True)
    eval_env.seed(seed + seed_offset)
    policy.actor.eval()

    avg_reward = 0.
    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False
        while not done:
            state = np.concatenate((state["observation"], state["desired_goal"]))
            state = (np.array(state).reshape(1, -1) - mean) / std
            action = policy.select_action(state)
            state, reward, done, _ = eval_env.step(action)
            avg_reward += reward
            # eval_env.render()
            time.sleep(0.05)

    avg_reward /= eval_episodes
    print(f"Avg reward {avg_reward}")
    return avg_reward

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # Experiment
    parser.add_argument("--policy", default="IQL")  # Policy name
    parser.add_argument("--data_path", default="/home/j/.d4rl/datasets")  # Path to data folder
    parser.add_argument("--env", default="PandaPush-v2")  # OpenAI gym environment name
    parser.add_argument("--seed", default=1, type=int)  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--eval_freq", default=1e4, type=int)  # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=1e6, type=int)  # Max time steps to run environment
    parser.add_argument("--save_model", action="store_true")  # Save model and optimizer parameters
    parser.add_argument("--load_model", default="IQL_PandaPush-v2_1_round0_finetuned")  # Model load file name, "" doesn't load, "default" uses file_name
    # IQL
    parser.add_argument("--batch_size", default=256, type=int)  # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99)  # Discount factor
    parser.add_argument("--tau", default=0.005)  # Target network update rate
    parser.add_argument("--expectile", default=0.7)  # Expectile parameter Tau
    parser.add_argument("--beta", default=3.0)  # Temperature parameter Beta
    parser.add_argument("--max_weight", default=100.0)  # Max weight for actor update
    parser.add_argument("--normalize_data", default=True)
    args = parser.parse_args()

    file_name = f"{args.policy}_{args.env}_{args.seed}"
    print("---------------------------------------")
    print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
    print("---------------------------------------")

    if not os.path.exists("./results"):
        os.makedirs("./results")

    if args.save_model and not os.path.exists("./models"):
        os.makedirs("./models")

    env = gym.make(args.env)

    # Set seeds
    env.seed(args.seed)
    env.action_space.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

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
    }

    # Initialize policy
    policy = IQL.IQL(**kwargs)

    if args.load_model != "":
        policy_file = file_name if args.load_model == "default" else args.load_model
        policy.load(f"./models/{policy_file}")

    eval_policy(policy, args.env, args.seed, 0, 1)