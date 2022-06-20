import numpy as np
from IPython import embed

data = np.load('offline_buffers/PandaPushv2_buffer.npz')
files_array = data.files
# print (files_array)
state_array = data['states']
action_array = data['actions']
reward_array = data['rewards']
done_array = data['dones']


done_true_idx = np.where(done_array == True)
desired_terminal_state = np.array([-0.1, 0.39,-0.38]) # desired object position
# desired_terminal_state = np.array([-0.15, 0.15, 0.02]) # desired object position
# desired_terminal_state = np.array([0, 0, 0.02]) # desired object position
done_true_idxs = done_true_idx[0]
start_idx = 0 # to keep track of the steps
r_max = 1 # should we modify this?
for done_idx in done_true_idxs:
    terminal_state = state_array[done_idx,6:9] #final object position
    # initial_state = state_array[start_idx, 6:9]

    l_final_to_goal = np.linalg.norm(desired_terminal_state - terminal_state)
    # # give the whole trajectory a reward based on whether the final state is closer to the desired goal than the initial state. 
    # l_initial_to_goal = np.linalg.norm(desired_terminal_state - initial_state)

    # r = r_max * ((l_initial_to_goal - l_final_to_goal) / l_initial_to_goal)
    # if (l_initial_to_goal > l_initial_to_goal)

    # interval = 50 // 3
    # for i in range(start_idx, done_idx - interval, interval):
    for i in range(start_idx, done_idx):
        # reward_array[i] = r
        current_state = state_array[i, 6:9] # current object position
        # next_state = state_array[i+interval, 6:9] # next object position

        # l1 = np.linalg.norm(desired_terminal_state - current_state) # norm between desired state and initial state
        # l2 = np.linalg.norm(desired_terminal_state - terminal_state) # norm between desired state and final state
        l_curr_to_goal = np.linalg.norm(desired_terminal_state - current_state)
        # l_next_to_goal = np.linalg.norm(desired_terminal_state - next_state)
        # # l_final_to_goal = np.linalg.norm(desired_terminal_state - terminal_state)

        if l_curr_to_goal <= 0.2:
            reward_array[i] = 0.
        else:
            reward_array[i] = -1. 

        # if (l_final_to_goal < 0.2): # The trajectory brought the object to the goal, so this sample gets some human reward
        #     r = r_max
        #     # reward_array[i] = r_max
        # elif (l_curr_to_goal - l_next_to_goal) < 0.005:
        #      # we're closer to the goal next state than this state. Give some small reward
        #     r = 0.
        #     # reward_array[i] = 0.
        # else:
        #     r = r_max*((l_final_to_goal-l_curr_to_goal)/l_curr_to_goal) 
        #     # reward_array[i] = r_max*((l_final_to_goal-l_curr_to_goal)/l_curr_to_goal) 

        # for j in range(i, i+interval):
        #     reward_array[j] = r
        # elif abs(l_curr_to_goal - l_next_to_goal) < 0.005: 
        #     reward_array[i] = -0.01 # slightly penalize staying still? 
        # # elif (l_final_to_goal < l_curr_to_goal): # We're closer to the goal than we started
        # #     reward_array[i] = r_max*((l_final_to_goal-l_curr_to_goal)/l_curr_to_goal) 
        # else: # otherwise normal negative reward
        #     reward_array[i] = -1


        # if (reward_array[i] > 0 or i%100000 == 0):
        #     print("======================")
        #     print(current_state)
        #     print(next_state)
        #     print(desired_terminal_state)
        #     print(reward_array[i])

    start_idx = done_idx+1

file_name = './offline_buffers/PandaPushv2_buffer_ood'
path_to_save =  file_name + '.npz'        

##
# file_name = 'PandaPushv2_buffer_no_reward'
# path_to_save =  file_name + '.npz'        
# print(f"ZEROING OUT ALL REWARD")
# reward_array[:] = 0
##


print(f"SAVING TO {path_to_save}")
hardcoded_desired_goals = np.ones(data["desired_goals"].shape) * desired_terminal_state 
np.savez_compressed(path_to_save, 
                    states = state_array, 
                    actions = action_array, 
                    rewards = reward_array, 
                    dones = done_array, 
                    achieved_goals = data['achieved_goals'], 
                    desired_goals = hardcoded_desired_goals)

