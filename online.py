import numpy as np
import torch
import gym
import argparse
import os
import d4rl_pybullet
import h5py
from tqdm import tqdm
import panda_gym
from torch.utils.tensorboard import SummaryWriter
import pickle
import utils
import IQL
from IPython import embed

# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, env_name, seed, mean, std, seed_offset=0, eval_episodes=10, render=False):
    eval_env = gym.make(env_name, render=render)
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

    avg_reward /= eval_episodes
    print(f"Avg reward {avg_reward}")
    return avg_reward


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # Experiment
    parser.add_argument("--policy", default="IQL")  # Policy name
    parser.add_argument("--data_path", default="/home/j/workspace/implicit-q-learning")  # Path to data folder
    parser.add_argument("--env", default="PandaPush-v2")  # OpenAI gym environment name
    parser.add_argument("--seed", default=1, type=int)  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--eval_freq", default=1e4, type=int)  # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=5e3, type=int)  # Max time steps to run environment
    parser.add_argument("--save_model", action="store_true")  # Save model and optimizer parameters
    parser.add_argument("--load_model", default="default")  # Model load file name, "" doesn't load, "default" uses file_name
    
    # IQL
    parser.add_argument("--batch_size", default=256, type=int)  # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99)  # Discount factor
    parser.add_argument("--tau", default=0.005)  # Target network update rate
    parser.add_argument("--expectile", default=0.8, type=float)  # Expectile parameter Tau
    parser.add_argument("--beta", default=3.0)  # Temperature parameter Beta
    parser.add_argument("--max_weight", default=100.0)  # Max weight for actor update
    parser.add_argument("--normalize_data", default=True)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--run_slot", default=-1, type=int)

    args = parser.parse_args()

    # for i in range(5)
    for i in [0]:
        comment = "ONLINE"
        comment += f"_slot{i}"
        comment += "_deterministic" if args.deterministic else ""
        writer = SummaryWriter(comment=comment)
        
        file_name = f"{args.policy}_{args.env}_{args.seed}_round{i}"
        save_file_name = file_name+"_ONLINE"
        print("---------------------------------------")
        print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
        print("---------------------------------------")

        if not os.path.exists("./results"):
            os.makedirs("./results")

        if args.save_model and not os.path.exists("./models"):
            os.makedirs("./models")

        env = gym.make(args.env)

        # embed()

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
            "finetuning": True,
            "deterministic_policy": args.deterministic
        }

        # Initialize policy
        policy = IQL.IQL(**kwargs)

        # In the case of D4RL-Pybullet dataset, else use conver_D4RL method
        # replay_buffer.convert_D4RL_pybullet(dataset)

        ## How do we normalize the data without using a replay buffer?
        # if args.normalize_data:
        #     mean, std = replay_buffer.calculate_meanstd()
        # else:
        mean, std = 0, 1

        n_epochs = max(1, int(args.max_timesteps) // int(args.eval_freq))
        # all_evaluations = dict()
        # evaluations = []
        eval = eval_policy(policy, args.env, args.seed, mean, std)
        print(f'initial eval: {eval}. Running for {n_epochs} epochs')

        '''
        Fill a new replay buffer with the last million samples of the old buffer
        '''
        replay_buffer = utils.ReplayBuffer(state_dim, action_dim)

        for epoch in range(n_epochs):
            range_gen = tqdm(
                range(int(args.eval_freq)),
                desc=f"Epoch {int(epoch)}/{n_epochs}",
            )
            mean_val = 0
            mean_q = 0
            mean_v_loss = 0
            mean_c_loss = 0
            mean_a_loss = 0

            state, done = env.reset(), False
            state = np.concatenate((state["observation"], state["desired_goal"]))
            for itr in range_gen:                
                # state are stored non-normalized. We pull the mean,std from the offline batch
                # normalized_state = (np.array(state).reshape(1, -1) - mean) / std
                # action = policy.select_action(normalized_state)
                action = policy.select_action(state)

                next_state, reward, done, _ = env.step(action)
                next_state = np.concatenate((next_state["observation"], next_state["desired_goal"]))

                replay_buffer.add(state, action, next_state, reward, done)                
                state = next_state

                if done:
                    state, done = env.reset(), False
                    state = np.concatenate((state["observation"], state["desired_goal"]))

                # if False:
                if len(replay_buffer) > args.batch_size:
                    # val, q, value_loss, critic_loss, actor_loss = policy.train(replay_buffer, args.batch_size, mean, std)
                    info = policy.train(replay_buffer, args.batch_size)
                    # print(info)
                    mean_val += info['value']
                    mean_q += info['q_val']
                    mean_c_loss += info['critic_loss']
                    mean_v_loss += info['value_loss']
                    mean_a_loss += info['actor_loss']

                    # for key, value in info.items():
                    #     all_evaluations[key] = all_evaluations.get(key, []) + [value]

            eval = eval_policy(policy, args.env, args.seed, mean, std, render=True if epoch == n_epochs-1 else False)
            
            writer.add_scalar('value/value', mean_val / len(range_gen), epoch)
            writer.add_scalar('value/qval', mean_q / len(range_gen), epoch)
            writer.add_scalar('Loss/value_loss', mean_v_loss / len(range_gen), epoch)
            writer.add_scalar('Loss/critic_loss', mean_c_loss / len(range_gen), epoch)
            writer.add_scalar('Loss/actor_loss', mean_a_loss / len(range_gen), epoch)
            writer.add_scalar("evaluations/eval", eval, epoch)
            # evaluations.append(eval)
            policy.actor_scheduler.step()
            print('Epoch {}/{}: value: {:.3f}. Q: {:.3f}. value_loss: {:.3f}. critic_loss: {:.3f}. actor_loss: {:.3f} env: {:.2f}'.format(
                                                                                            epoch, n_epochs,
                                                                                            mean_val / len(range_gen),
                                                                                            mean_q / len(range_gen),
                                                                                            mean_v_loss / len(range_gen),
                                                                                            mean_c_loss / len(range_gen),
                                                                                            mean_a_loss / len(range_gen),
                                                                                            eval))
            
            print(f"Writing to file {save_file_name}")
            
            # with open(f"./results/{save_file_name}.pickle", "wb") as f:
            #     pickle.dump(evaluations, f)
            # with open(f"./results/all_{save_file_name}.pickle", "wb") as f:
            #     pickle.dump(all_evaluations, f)

            # np.save(f"./results/{file_name}", evaluations)
            # np.save(f"./results/all_{file_name}", all_evaluations, allow_pickle=True)
            if args.save_model: policy.save(f"./models/{save_file_name}")