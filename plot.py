from torch.utils.tensorboard import SummaryWriter
import numpy as np
import pickle
from IPython import embed
import matplotlib.pyplot as plt
import pandas as pd
import argparse

epoch_scores_agg = []
aggregate = dict()


parser = argparse.ArgumentParser()
parser.add_argument("--finetuned", action="store_true")
parser.add_argument("--notes", default="")  # Model load file name, "" doesn't load, "default" uses file_name
parser.add_argument("--no_training", action="store_true")
args = parser.parse_args()

INCLUDE_FINETUNED = args.finetuned

n_runs = 3

# for i in range(n_runs):
for i in [2,3]:
    epoch_fn = f"./results/IQL_PandaPush-v2_1_round{i}.pickle"
    all_evals_fn = f"./results/all_IQL_PandaPush-v2_1_round{i}.pickle"
    print(f"Adding results from {epoch_fn}")

    with open(epoch_fn, 'rb') as f:
        epoch_scores = pickle.load(f)

    with open(all_evals_fn, 'rb') as f:
        all_evaluations = pickle.load(f)

    
    if (INCLUDE_FINETUNED):
        epoch_fn = f"./results/IQL_PandaPush-v2_1_round{i}_finetuned.pickle"
        all_evals_fn = f"./results/all_IQL_PandaPush-v2_1_round{i}_finetuned.pickle"

        print(f"   Adding finetuned results from {epoch_fn}")

        with open(epoch_fn, 'rb') as f:
            epoch_scores_finetuned = pickle.load(f)

        with open(all_evals_fn, 'rb') as f:
            all_evaluations_finetuned = pickle.load(f)

        epoch_scores = np.concatenate((epoch_scores, epoch_scores_finetuned))

    epoch_scores_agg.append(np.array(epoch_scores))

    for key, value in all_evaluations.items():
        aggregate[key] = aggregate.get(key, []) + [np.array(value)]


print(f"Found {len(epoch_scores_agg)} runs")
epoch_scores_agg = np.array(epoch_scores_agg)

def plot_with_stdv(data, xlabel, ylabel, title, use_ma = False, note=""):
    robot_data_avg = np.average(data, axis=0)
    robot_stdv = np.std(data, axis=0)
    
    if (use_ma): # moving average
        print(f"Using moving average")
        window_size = 50
        i = 0
        # Initialize an empty list to store moving averages
        moving_averages = []
        
        # Loop through the array t o
        #consider every window of size 3
        while i < len(robot_data_avg) - window_size + 1:
        
            # Calculate the average of current window
            window_average = round(np.sum(robot_data_avg[
            i:i+window_size]) / window_size, 2)
            
            # Store the average of current
            # window in moving average list
            moving_averages.append(window_average)
            
            # Shift window to right by one position
            i += 1
        robot_data_avg = np.array(moving_averages)
        robot_stdv = robot_stdv[window_size-1:]

    fig, ax = plt.subplots() 
    right_side = ax.spines["right"]
    top_side = ax.spines["top"]
    right_side.set_visible(False)
    top_side.set_visible(False)

    # First we plot the lines:
    plt.plot(robot_data_avg, label="Robot", linewidth=2) 
    # plt.plot(human_data_avg, label="Human", linewidth=2)

    # Then we plot the fill
    plt.fill_between([i for i in range(len(robot_data_avg))], (robot_data_avg-robot_stdv), (robot_data_avg+robot_stdv), alpha=0.3)
    # plt.fill_between([i for i in range(len(human_data_avg))], (human_data_avg-human_stdv), (human_data_avg+human_stdv), alpha=0.3)
    # Here alpha dicated the opacity of the fill.

    plt.yticks(fontsize=16) 
    plt.xticks(fontsize=16) 
    
    plt.legend(fontsize="xx-large")

    plt.xlabel(xlabel, fontsize="xx-large")
    plt.ylabel(ylabel, fontsize="xx-large")
    plt.title(title, fontsize="xx-large")

    fn = f"./plots/{title}_{note}.png"
    print(f"  writing plot to {fn}")
    plt.savefig(fn)
    # plt.cla()
plt.show()

note = "finetuned" if INCLUDE_FINETUNED else "offline"
# embed()
plot_with_stdv(epoch_scores_agg, "epoch", "Average Reward", f"epoch_rewards_n{n_runs}", note=note)
if (INCLUDE_FINETUNED):
    plot_with_stdv(epoch_scores_finetuned, "epoch", "Average Reward", f"fintuned_epoch_rewards_n{n_runs}", note=note+args.notes)

# for key, value in aggregate.items():
#     data = np.array(value)
#     plot_with_stdv(np.array(value), "Step", "#", key, use_ma=True, note=note)