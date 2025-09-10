import numpy as np
import copy
import gymnasium
from gymnasium import spaces

class VMAPDWrapper(object):

    def __init__(self, env, max_z, fix_z):
        super().__init__()
        self.env = env
        self.max_z = max_z
        self.fix_z = fix_z
        self.cur_z = -1
        self.num_agents = 1
        self.z_space = [spaces.Discrete(self.max_z) for _ in range(self.num_agents)]
        self.z_obs_space = [copy.copy(self.env.observation_space[0])]
        self.z_local_obs_space = [copy.copy(self.env.observation_space[0])]
        self.observation_space = [copy.copy(self.env.observation_space[0])]
        self.share_observation_space = [copy.copy(self.env.observation_space[0])]
        self.action_space = self.env.action_space

        # Create new expanded observation spaces
        old_obs_space = self.env.observation_space[0]
        new_obs_shape = (old_obs_space.shape[0] + self.max_z,)
        new_obs_space = spaces.Box(
            low=old_obs_space.low.min(),
            high=old_obs_space.high.max(),
            shape=new_obs_shape,
            dtype=old_obs_space.dtype
        )

        # Replace the observation spaces with expanded versions
        self.observation_space = [new_obs_space]
        self.share_observation_space = [new_obs_space]

#Original code, outdated and doesnt work 
        # for observation_space in self.observation_space:
        #     observation_space.shape = (observation_space.shape[0] + self.max_z,)
        # for observation_space in self.share_observation_space:
        #     observation_space.shape = (observation_space.shape[0] + self.max_z,)



#original with dim issue
    # def reset(self, fix_z=None):
    #     if fix_z is not None:
    #         self.cur_z = fix_z
    #     elif self.fix_z is not None:
    #         self.cur_z = self.fix_z
    #     else:
    #         self.cur_z = np.random.randint(self.max_z) 
    #     obs_n = self.env.reset()
    #     z_vec = np.eye(self.max_z)[self.cur_z]
    #     z_vec = np.expand_dims(z_vec, 0)
    #     z_vec = z_vec.repeat(self.num_agents, 0)
    #     obs_n = np.concatenate([z_vec, np.array(obs_n)], -1)
    #     return obs_n

    def reset(self, fix_z=None):
        if fix_z is not None:
            self.cur_z = fix_z
        elif self.fix_z is not None:
            self.cur_z = self.fix_z
        else:
            self.cur_z = np.random.randint(self.max_z)
        
        obs_n = self.env.reset()
        
        z_vec = np.eye(self.max_z)[self.cur_z]
        z_vec = np.expand_dims(z_vec, 0)
        z_vec = z_vec.repeat(self.num_agents, 0)
        
        # Ensure z_vec has the right shape for concatenation
        if z_vec.ndim == 3:
            z_vec = z_vec.squeeze(1)  # Remove the extra middle dimension
        
        
        obs_n = np.concatenate([z_vec, np.array(obs_n)], -1)
        return obs_n
        


#original with dim issue
    # def step(self, actions):
    #     # obs_n, reward_n, done_n, info_n = self.env.step(actions)

    #     print(f"DEBUG: VMAPD calling env.step with actions: {actions}")
    #     step_result = self.env.step(actions)
    #     print(f"DEBUG: VMAPD received result length: {len(step_result)}")
    #     print(f"DEBUG: VMAPD result types: {[type(x) for x in step_result]}")
        
    #     if len(step_result) == 4:
    #         obs_n, reward_n, done_n, info_n = step_result
    #     else:
    #         obs_n, reward_n, terminated, truncated, info_n = step_result
    #         done_n = terminated or truncated


    #     z_vec = np.eye(self.max_z)[self.cur_z]
    #     z_vec = np.expand_dims(z_vec, 0)
    #     z_vec = z_vec.repeat(self.num_agents, 0)
    #     obs_n = np.concatenate([z_vec, np.array(obs_n)], -1)
    #     return obs_n, reward_n, done_n, info_n

#new to fix dim issue
    def step(self, action):
        obs_n, rewards, dones, infos = self.env.step(action)
        
        z_vec = np.eye(self.max_z)[self.cur_z]
        z_vec = np.expand_dims(z_vec, 0)
        z_vec = z_vec.repeat(self.num_agents, 0)
        
        # Add the same fix here
        if z_vec.ndim == 3:
            z_vec = z_vec.squeeze(1)  # Remove the extra middle dimension
        
        obs_n = np.concatenate([z_vec, np.array(obs_n)], -1)
        return obs_n, rewards, dones, infos




    def seed(self, seed):

        try:
            return self.env.seed(seed)
        except AttributeError:
            # Some wrappers in the chain don't have seed() - that's okay
            # The important thing is that we tried to seed
            # print(f"Warning: Could not seed environment, but continuing...")
            return [seed]
 
        


    def close(self):
        self.env.close()

    def render(self, mode='human'):
        img = self.env.render(mode)
        return img