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
from utils import eval_policy
from IPython import embed

def online_finetuning(args, buffer_name, load_model_path=None):
    comment = f"{args.comment}_onlineFT_{buffer_name}"
    writer = SummaryWriter(comment=comment)
    file_name = f"{comment}"
    print("---------------------------------------")
    print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
    print("---------------------------------------")

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
        "deterministic_policy": args.deterministic,
    }

    # Initialize policy
    policy = IQL.IQL(**kwargs)
    if load_model_path is not None:
        print(f"\tLoading policy from {load_model_path}")
        policy.load(load_model_path)

    # Load the offline data to get the mean and std we've previously trained under
    
    replay_buffer = utils.ReplayBuffer(state_dim, action_dim)
    if buffer_name:
        offline_dataset_path = os.path.join(args.data_path, buffer_name+".npz")
        print(f"\tLoading offline data from {offline_dataset_path}")
        dataset = np.load(offline_dataset_path)
        replay_buffer.convert_npz(dataset)
        print(f"\t\tOffline dataset size: {len(replay_buffer)}")
        args.normalize_data = False

    if args.normalize_data:
        mean, std = replay_buffer.normalize_states()
    else:
        print(f"Not normalizing data.")
        mean, std = 0, 1

    n_epochs = max(1, int(args.max_timesteps) // int(args.eval_freq))

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
                info = policy.train(replay_buffer, args.batch_size)
                mean_val += info['value']
                mean_q += info['q_val']
                mean_c_loss += info['critic_loss']
                mean_v_loss += info['value_loss']
                mean_a_loss += info['actor_loss']

        eval = eval_policy(policy, args.env, args.seed, mean, std)        
        writer.add_scalar('value/value', mean_val / len(range_gen), epoch)
        writer.add_scalar('value/qval', mean_q / len(range_gen), epoch)
        writer.add_scalar('Loss/value_loss', mean_v_loss / len(range_gen), epoch)
        writer.add_scalar('Loss/critic_loss', mean_c_loss / len(range_gen), epoch)
        writer.add_scalar('Loss/actor_loss', mean_a_loss / len(range_gen), epoch)
        writer.add_scalar("evaluations/eval", eval, epoch)
        policy.actor_scheduler.step()
        print('Epoch {}/{}: value: {:.3f}. Q: {:.3f}. value_loss: {:.3f}. critic_loss: {:.3f}. actor_loss: {:.3f} env: {:.2f}'.format(
                                                                                        epoch, n_epochs,
                                                                                        mean_val / len(range_gen),
                                                                                        mean_q / len(range_gen),
                                                                                        mean_v_loss / len(range_gen),
                                                                                        mean_c_loss / len(range_gen),
                                                                                        mean_a_loss / len(range_gen),
                                                                                        eval))
        
        
        print(f"Writing to file {file_name}")
        if args.save_model: policy.save(f"./models/{file_name}")
    policy.save("./models/last")


if __name__ == "__main__":
    import argparse
    import datetime
    parser = argparse.ArgumentParser()
    # Experiment
    parser.add_argument("--policy", default="IQL")  # Policy name
    parser.add_argument("--data_path", default="/home/j/workspace/implicit-q-learning/offline_buffers")
    parser.add_argument("--buffer", default="PandaPushv2_buffer")
    parser.add_argument("--env", default="PandaPush-v2")  # OpenAI gym environment name
    parser.add_argument("--mod", action="store_true")
    parser.add_argument("--seed", default=1, type=int)  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--eval_freq", default=1e4, type=int)  # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=2e5, type=int)  # Max time steps to run environment
    parser.add_argument("--save_model", default=True)  # Save model and optimizer parameters
    parser.add_argument("--load_model", default="")  # Model load file name, "" doesn't load, "default" uses file_name
    # IQL
    parser.add_argument("--batch_size", default=256, type=int)  # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99)  # Discount factor
    parser.add_argument("--tau", default=0.005)  # Target network update rate
    parser.add_argument("--expectile", default=0.8)  # Expectile parameter Tau
    parser.add_argument("--beta", default=3.0)  # Temperature parameter Beta
    parser.add_argument("--max_weight", default=100.0)  # Max weight for actor update
    parser.add_argument("--normalize_data", default=True)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--run_slot", type=int, default=-1)
    parser.add_argument("--comment", default="")
    args = parser.parse_args()

    if (args.mod):
        args.env = "PandaPushModified-v2"
        args.comment += "MOD"

    datestring = f"{datetime.datetime.now().strftime('%m-%d_%H-%M-%S')}"
    args.comment += f"{datestring}"

    online_finetuning(args, args.buffer, None)