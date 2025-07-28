import gymnasium as gym
import numpy as np
import cv2


class AtariPreprocessing(gym.ObservationWrapper):
    """Atari preprocessing wrapper that converts images to grayscale and resizes them."""
    
    def __init__(self, env, frame_size=84, grayscale=True):
        super().__init__(env)
        self.frame_size = frame_size
        self.grayscale = grayscale
        
        if grayscale:
            self.observation_space = gym.spaces.Box(
                low=0, high=255, 
                shape=(frame_size * frame_size,), 
                dtype=np.uint8
            )
        else:
            self.observation_space = gym.spaces.Box(
                low=0, high=255, 
                shape=(frame_size * frame_size * 3,), 
                dtype=np.uint8
            )
    
    def observation(self, obs):
        # Convert to grayscale if specified
        if self.grayscale:
            obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        
        # Resize the frame
        obs = cv2.resize(obs, (self.frame_size, self.frame_size), interpolation=cv2.INTER_AREA)
        
        # Flatten the observation
        obs = obs.flatten()
        
        return obs.astype(np.float32) / 255.0  # Normalize to [0, 1]


class NoOpResetEnv(gym.Wrapper):
    """Sample initial states by taking random number of no-ops on reset."""
    
    def __init__(self, env, noop_max=30):
        gym.Wrapper.__init__(self, env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.unwrapped.np_random.integers(1, self.noop_max + 1)
        assert noops > 0
        for _ in range(noops):
            obs, _, terminated, truncated, _ = self.env.step(self.noop_action)
            if terminated or truncated:
                obs, info = self.env.reset(**kwargs)
        return obs, info


def make_atari_env(env_name, frame_size=84, noop_max=30):
    """Create an Atari environment with standard preprocessing."""
    env = gym.make(env_name)
    env = NoOpResetEnv(env, noop_max=noop_max)
    env = AtariPreprocessing(env, frame_size=frame_size, grayscale=True)
    return env 