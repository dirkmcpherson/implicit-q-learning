from tbparse import SummaryReader
import os, sys
import pandas as pd
from IPython import embed
import numpy as np

path = "./runs/"
dirs = os.listdir(path)

keys = ["NORMREG", "NORMNOREG", "MODREG", "MODNOREG"]
series = {key:[] for key in keys}

# tag = "evaluations/eval"

for logdir in dirs:
    print(f"\tReading {logdir}")
    reader = SummaryReader(path+logdir)


    # embed()
    # sys.exit()
    df = reader.scalars
    tags = df["tag"].unique()
    # df = df[df["tag"] == tag]
    
    for key in keys:
        if key in logdir:
            series[key].append(df)
            break

import matplotlib.pyplot as plt
for tag in tags:
    fig, ax = plt.subplots() 

    right_side = ax.spines["right"]
    top_side = ax.spines["top"]
    right_side.set_visible(False)
    top_side.set_visible(False)

    style_idx = 0
    for key in keys:
        dfs = series[key]
        ys = np.array([df[df["tag"] == tag]["value"].values for df in dfs])
        
        data_avg, data_std = np.mean(ys, axis=0), np.std(ys, axis=0)

        if (style_idx % 2 == 0):
            plt.plot(data_avg, label=key, linewidth=2) 
        else:
            plt.plot(data_avg, label=key, linewidth=2, linestyle='--')
        style_idx += 1
        plt.fill_between([i for i in range(len(data_avg))], (data_avg-data_std), (data_avg+data_std), alpha=0.3)

    plt.yticks(fontsize=16) 
    plt.xticks(fontsize=16) 

    plt.legend(fontsize="xx-large")

    plt.xlabel("step", fontsize="xx-large")
    plt.ylabel("Average value", fontsize="xx-large")
    plt.title(f"{tag}", fontsize="xx-large")

    plotdir = "./plots/agg/"
    if not os.path.exists(plotdir): os.makedirs(plotdir)
    plt.savefig(plotdir+f"{tag.replace('/', '_')}", bbox_inches="tight") # The bbox_inches="tight" ensures all of our graph is saved!
    plt.show()
    
    # log_dir = "./runs/Jul02_09-59-41_pop-osMOD07-02_09-59-41_onlineFT_PandaPushv2_buffer"
    # reader = SummaryReader(log_dir)
    # df = reader.scalars
    # print(df)