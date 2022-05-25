import numpy as np
from IPython import embed

data = np.load('PandaPushv2_buffer.npz')
files_array = data.files
# print (files_array)
state_array = data['states']
action_array = data['actions']
reward_array = data['rewards']
done_array = data['dones']


done_true_idx = np.where(done_array == True)
desired_terminal_state = np.array([-1.07467088e-01,  4.82145741e-01, -3.80011010e-01]) # desired object position
done_true_idxs = done_true_idx[0]
start_idx = 0 # to keep track of the steps
r_max = 1 # should we modify this?
for done_idx in done_true_idxs:
    terminal_state = state_array[done_idx,6:9] #final object position
    for i in range(start_idx, done_idx):
        current_state = state_array[i, 6:9] # current object position
        terminal_state = state_array[done_idx, 6:9] # final object position
        l1 = np.linalg.norm(desired_terminal_state - current_state) # norm between desired state and initial state
        l2 = np.linalg.norm(desired_terminal_state - terminal_state) # norm between desired state and final state

        if (l2 < 0.1): # The trajectory brought the object to the goal
            reward_array[i] = r_max
        # elif (l1-l2 == 0): # 0 reward for no change?  
        #     reward_array[i] = 0
        elif (l2 < l1): # We're closer to the goal than we started
        # else:
            reward_array[i] = r_max*((l1-l2)/l1) 
        else: # otherwise normal negative reward
            reward_array[i] = -1
    start_idx = done_idx+1



file_name = 'PandaPushv2_buffer_modified'
path_to_save =  file_name + '.npz'        
# np.savez_compressed(path_to_save, states = state_array, actions = action_array, rewards = reward_array, dones = done_array, achieved_goals = data['achieved_goals'], desired_goals = data['desired_goals'])


hardcoded_desired_goals = np.ones(data["desired_goals"].shape) * desired_terminal_state 
np.savez_compressed(path_to_save, 
                    states = state_array, 
                    actions = action_array, 
                    rewards = reward_array, 
                    dones = done_array, 
                    achieved_goals = data['achieved_goals'], 
                    desired_goals = hardcoded_desired_goals)

