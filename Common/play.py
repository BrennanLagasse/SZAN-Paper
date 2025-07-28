# from mujoco_py.generated import const
#from mujoco_py import GlfwContext
import cv2
import numpy as np
import os

#GlfwContext(offscreen=True)


class Play:
    def __init__(self, env, agent, n_skills):
        self.env = env
        self.agent = agent
        self.n_skills = n_skills
        self.agent.set_policy_net_to_cpu_mode()
        self.agent.set_policy_net_to_eval_mode()
        self.fourcc = cv2.VideoWriter_fourcc(*'XVID')
        if not os.path.exists("Vid/"):
            os.mkdir("Vid/")

    @staticmethod
    def concat_state_latent(s, z_, n):
        z_one_hot = np.zeros(n)
        z_one_hot[z_] = 1
        return np.concatenate([s, z_one_hot])

    def evaluate(self):

        for z in range(self.n_skills):
            video_writer = cv2.VideoWriter(f"Vid/skill{z}" + ".avi", self.fourcc, 50.0, (250, 250))
            s, _ = self.env.reset()  # Gymnasium API change
            s = self.concat_state_latent(s, z, self.n_skills)
            episode_reward = 0
            max_steps = self.env.spec.max_episode_steps if self.env.spec.max_episode_steps else 1000
            for _ in range(max_steps):
                action = self.agent.choose_action(s)
                s_, r, terminated, truncated, _ = self.env.step(action)  # Gymnasium API change
                done = terminated or truncated
                s_ = self.concat_state_latent(s_, z, self.n_skills)
                episode_reward += r
                if done:
                    break
                s = s_
                # Need to create a temporary environment with render_mode for video
                try:
                    # Try to get the underlying environment and render
                    base_env = self.env
                    while hasattr(base_env, 'env'):
                        base_env = base_env.env
                    
                    if hasattr(base_env, 'ale'):
                        # For ALE environments, get the screen directly
                        I = base_env.ale.getScreenRGB()
                    else:
                        # Fallback - create a new env instance for rendering
                        import gymnasium as gym
                        import ale_py
                        gym.register_envs(ale_py)
                        from Common.atari_wrappers import make_atari_env
                        render_env = gym.make(self.env.spec.id, render_mode='rgb_array')
                        I = render_env.render()
                        render_env.close()
                        
                    if I is not None:
                        I = cv2.cvtColor(I, cv2.COLOR_RGB2BGR)
                        I = cv2.resize(I, (250, 250))
                        video_writer.write(I)
                except Exception as e:
                    print(f"Warning: Could not render frame: {e}")
                    
            print(f"skill: {z}, episode reward:{episode_reward:.1f}")
            video_writer.release()
        self.env.close()
        cv2.destroyAllWindows()
