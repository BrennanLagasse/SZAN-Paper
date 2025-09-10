import time
import numpy as np
from collections import deque
from onpolicy.runner.shared.base_runner import Runner
import torch

class BaselinePongRunner(Runner):
    """Simplified PPO runner for Pong without DGPO - for debugging environment issues"""
    
    def __init__(self, config):
        super(BaselinePongRunner, self).__init__(config)
        self.episode_rewards = []
        self.episode_count = 0
        self.total_episodes_completed = 0
        
        # Simple reward tracking
        self.env_rewards = [[] for _ in range(self.n_rollout_threads)]
        self.completed_rewards = deque(maxlen=100)

    def run(self):
        self.warmup()
        
        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads
        
        for episode in range(episodes):
            self.episode_count = episode
            if self.episode_count % 5 == 0:
                print(f"Baseline PPO Episode: {self.episode_count}")
                
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)
                
            for step in range(self.episode_length):
                # Collect standard PPO data (no DGPO components)
                values, actions, action_log_probs, rnn_states, rnn_states_critic = self.collect_baseline(step)
                
                # Environment step
                obs, rewards, dones, infos = self.envs.step(actions)
                
                # Track rewards for episode completion
                for i in range(self.n_rollout_threads):
                    reward_scalar = rewards[i][0][0]
                    self.env_rewards[i].append(reward_scalar)
                
                # Insert standard PPO data (no z_log_probs or dual critics)
                data = dict()
                data['obs'] = obs
                data['share_obs'] = obs.copy()
                data['rnn_states_actor'] = rnn_states
                data['rnn_states_critic'] = rnn_states_critic
                data['actions'] = actions
                data['action_log_probs'] = action_log_probs
                data['value_preds'] = values
                data['rewards'] = rewards  # Clean rewards only
                data['dones'] = dones
                self.insert(data, step)
                
                # Episode completion tracking
                if infos is not None:
                    for i, info in enumerate(infos):
                        if dones[i].any():
                            episode_length = len(self.env_rewards[i])
                            episode_reward = sum(self.env_rewards[i])
                            
                            self.episode_rewards.append(episode_reward)
                            self.completed_rewards.append(episode_reward)
                            self.total_episodes_completed += 1
                            
                            print(f"Baseline Episode {self.total_episodes_completed}: Length={episode_length}, Reward={episode_reward:.2f}")
                            
                            # Reset tracking
                            self.env_rewards[i] = []
            
            # Standard PPO training (no dual critics or discriminator)
            self.compute()
            train_infos = self.train()
            
            # Simple logging
            if self.episode_count % 10 == 0:
                if train_infos:
                    print(f"Training - Policy Loss: {train_infos.get('policy_loss', 0):.4f}, Value Loss: {train_infos.get('value_loss', 0):.4f}")
                
                if len(self.completed_rewards) > 0:
                    recent_rewards = list(self.completed_rewards)[-10:]  # Last 10 episodes
                    print(f"Recent performance: Mean={np.mean(recent_rewards):.2f}, Best={np.max(recent_rewards):.2f}")
            
            # Standard post-processing
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads
            
            if (episode % self.save_interval == 0 or episode == episodes - 1):
                self.save()
                
            if episode % self.log_interval == 0:
                end = time.time()
                if train_infos:
                    train_infos["FPS"] = int(total_num_steps / (end - start))
                    if len(self.episode_rewards) > 0:
                        train_infos["episode_rewards_mean"] = np.mean(self.episode_rewards)
                self.log_train(train_infos, total_num_steps)

    def collect_baseline(self, step):
        """Simplified collect method - standard PPO without DGPO components"""
        self.trainer.prep_rollout()
        
        value, action, action_log_prob, rnn_state, rnn_state_critic = \
            self.trainer.policy.get_actions(
                self.buffer.share_obs[step],
                self.buffer.obs[step], 
                self.buffer.rnn_states[step],
                self.buffer.rnn_states_critic[step],
                self.buffer.masks[step]
            )
            
        # Convert actions to environment format
        actions_env = action.copy()
        
        return value, action, action_log_prob, rnn_state, rnn_state_critic

    def warmup(self):
        """Simple warmup without DGPO components"""
        # Reset envs
        obs = self.envs.reset()
        
        # Initialize buffer with standard PPO structure (no dual critics)
        self.buffer.obs[0] = obs.copy()
        self.buffer.share_obs[0] = obs.copy()

    @torch.no_grad()
    def render(self):
        """Standard render for Pong"""
        envs_to_render = [0]  # Render first environment
        for env_i in envs_to_render:
            rendered_image = self.envs.render(mode='rgb_array')[env_i]
            # Save or display the rendered image as needed