import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np
from IPython import embed


data = np.load('PandaPushv2_buffer_modified_IIII.npz')
files_array = data.files
state_array = data['states']
action_array = data['actions']
reward_array = data['rewards']
done_array = data['dones']

done_true_idx = np.where(done_array == True)
desired_terminal_state = np.array([-0.1, 0.39,-0.38]) # desired object position
done_true_idxs = done_true_idx[0]
start_idx = 0 # to keep track of the steps
r_max = 1 # should we modify this?
for done_idx in done_true_idxs:
    path = []
    rewards = []
    sizes = []
    size = 1
    for i in range(start_idx, done_idx):
        path.append(state_array[i, 6:9])
        rewards.append(reward_array[i])
        sizes.append(size)
        size += 1
    start_idx = done_idx+1

    # fig = plt.figure()
    # ax = plt.axes(projection='3d')
    # Data for three-dimensional scattered points
    xdata = [pt[0] for pt in path]
    ydata = [pt[1] for pt in path]
    # zdata = [pt[2] for pt in path]
    # ax.scatter3D(xdata, ydata, zdata, c=rewards, cmap='Greens');
    # ax.scatter3D(*desired_terminal_state, marker='^')
    print(xdata)
    print(ydata)
    print(rewards)
    print(sizes)
    plt.scatter(xdata, ydata, marker='o', c=rewards, cmap="RdYlBu", s=sizes)
    plt.scatter(desired_terminal_state[0], desired_terminal_state[1], marker='^')
    plt.colorbar()
    plt.show()
    plt.cla()