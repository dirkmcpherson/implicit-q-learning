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
from IPython import embed

# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, env_name, seed, mean, std, seed_offset=0, eval_episodes=10, render=False):
    eval_env = gym.make(env_name, render=render)
    eval_env.seed(seed + seed_offset)
    policy.actor.eval()

    avg_reward = 0.
    all_eps_rewards = []
    for _ in range(eval_episodes):
        ep_reward = 0
        state, done = eval_env.reset(), False
        while not done:
            state = np.concatenate((state["observation"], state["desired_goal"]))
            state = (np.array(state).reshape(1, -1) - mean) / std
            action = policy.select_action(state)
            state, reward, done, _ = eval_env.step(action)
            # print_state = np.concatenate((state["observation"], state["desired_goal"]))
            # print(f"{state} {action} {reward} {done}")
            avg_reward += reward
            ep_reward += reward
            if render:
                time.sleep(0.05)
        all_eps_rewards.append(ep_reward)
        # print(f"eval ep reward {ep_reward}")

    avg_reward /= eval_episodes
    print(f"Avg reward {avg_reward} std {np.std(all_eps_rewards)}")
    return avg_reward, all_eps_rewards

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # Experiment
    parser.add_argument("--policy", default="IQL")  # Policy name
    parser.add_argument("--data_path", default="/home/j/workspace/implicit-q-learning")  # Path to data folder
    parser.add_argument("--env", default="PandaPush-v2")  # OpenAI gym environment name
    parser.add_argument("--seed", default=1, type=int)  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--eval_freq", default=1e4, type=int)  # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=1e6, type=int)  # Max time steps to run environment
    parser.add_argument("--save_model", action="store_true")  # Save model and optimizer parameters
    parser.add_argument("--load_model", default="IQL_PandaPush-v2_1_round2")  # Model load file name, "" doesn't load, "default" uses file_name
    # IQL
    parser.add_argument("--batch_size", default=256, type=int)  # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99)  # Discount factor
    parser.add_argument("--tau", default=0.005)  # Target network update rate
    parser.add_argument("--expectile", default=0.7)  # Expectile parameter Tau
    parser.add_argument("--beta", default=3.0)  # Temperature parameter Beta
    parser.add_argument("--max_weight", default=100.0)  # Max weight for actor update
    parser.add_argument("--normalize_data", default=True)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--finetuned", action="store_true")
    parser.add_argument("--eval_episodes", default=100, type=int)
    parser.add_argument("--deterministic", action="store_true")
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

    replay_buffer = utils.ReplayBuffer(state_dim, action_dim)
    dataset = np.load(os.path.join(args.data_path, 'PandaPushv2_buffer.npz'))
    replay_buffer.convert_npz(dataset)
    if args.normalize_data:
        mean, std = replay_buffer.normalize_states()
    else:
        mean, std = 0, 1

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
        "deterministic_policy": args.deterministic
    }



    # Initialize policy
    policy = IQL.IQL(**kwargs)

    if args.load_model != "":
        policy_file = file_name if args.load_model == "default" else args.load_model
        # embed()
        if (args.finetuned): policy_file = policy_file + "_finetuned"
        print(f"Loading policy from {policy_file}")
        policy.load(f"./models/{policy_file}")

    avg_rew, all_ep_rewards = eval_policy(policy, args.env, args.seed, mean, std, eval_episodes=args.eval_episodes, render=args.render)

    import matplotlib.pyplot as plt
    import numpy as np
    import scipy.stats as stats
    import math

    mu = np.mean(all_ep_rewards)
    variance = np.var(all_ep_rewards)
    sigma = math.sqrt(variance)
    x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
    title = "finetuned_offline" if args.finetuned else "offline"
    plt.title(title)
    plt.plot(x, stats.norm.pdf(x, mu, sigma))
    plt.savefig(f"./plots/{title}.png")
    plt.show()