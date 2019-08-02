import numpy as np


def generate_environment(env_size, obs_val, rew_val):

    environment = np.zeros((env_size, env_size), dtype='int32')
    obstacles = np.zeros((env_size, env_size), dtype='int32')

    mid_pos = int(env_size/2) 

    inds = np.arange(env_size)
    col_inds =  [mid_pos-1, mid_pos, mid_pos+1] #inds[0:mid_pos -2 ]  
    row_inds =  inds[len(inds)- mid_pos + 2: len(inds)]

    # fill the cols
    for i in row_inds:
        for ii in col_inds:
            environment[i, ii] = obs_val  
            environment[ii, i] = obs_val
            environment[ii, env_size - i -1] = obs_val
            environment[env_size - i -1, ii] = obs_val

    environment[0,0] = rew_val
    environment[env_size-1, env_size-1] = rew_val
    environment[0, env_size-1] = rew_val
    environment[env_size-1, 0] = rew_val


    return environment
