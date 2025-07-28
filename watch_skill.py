#!/usr/bin/env python3
"""
Watch a specific DIAYN skill play Pong with visual rendering
Usage: python watch_skill.py 3 1  (skill_id episodes)
"""
import gymnasium as gym
import ale_py
import numpy as np
import sys
from Brain import SACAgent
from Common import Logger, make_atari_env

def concat_state_latent(s, z_, n):
    z_one_hot = np.zeros(n)
    z_one_hot[z_] = 1
    return np.concatenate([s, z_one_hot])

def watch_skill():
    # Simple argument parsing to avoid conflicts
    if len(sys.argv) < 2:
        skill_id = 3  # Default to best skill
        episodes = 1
        print("Usage: python watch_skill.py <skill_id> <episodes>")
        print(f"Using defaults: skill_id={skill_id}, episodes={episodes}")
    else:
        skill_id = int(sys.argv[1]) if len(sys.argv) > 1 else 3
        episodes = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    
    n_skills = 10  # Fixed for our current setup
    
    print(f"🎮 Watching Skill {skill_id} play Pong for {episodes} episodes...")
    
    # Load parameters with empty sys.argv to avoid conflicts
    original_argv = sys.argv[:]
    sys.argv = ['watch_skill.py']  # Clear arguments temporarily
    
    try:
        from Common.config import get_params
        params = get_params()
    finally:
        sys.argv = original_argv  # Restore arguments
    
    gym.register_envs(ale_py)
    
    # Create environment with human rendering
    try:
        env = gym.make(params["env_name"], render_mode='human')
        print("✅ Visual rendering enabled")
    except Exception as e:
        print(f"⚠️  Visual rendering failed: {e}")
        print("   Falling back to analysis mode...")
        env = make_atari_env(params["env_name"])
    
    # Create the processed environment for the agent
    agent_env = make_atari_env(params["env_name"])
    n_states = agent_env.observation_space.shape[0]
    n_actions = agent_env.action_space.n
    action_bounds = [0, agent_env.action_space.n - 1]

    params.update({"n_states": n_states,
                   "n_actions": n_actions,
                   "action_bounds": action_bounds,
                   "n_skills": n_skills})

    # Create agent and load weights
    p_z = np.full(params["n_skills"], 1 / params["n_skills"])
    agent = SACAgent(p_z=p_z, **params)
    logger = Logger(agent, **params)
    
    try:
        logger.load_weights()
        print("✅ Successfully loaded trained weights!")
    except Exception as e:
        print(f"❌ Could not load weights: {e}")
        return
    
    # Set agent to evaluation mode
    agent.policy_network.eval()
    
    print(f"\n🤖 Skill {skill_id} Performance Preview:")
    print("="*50)
    
    total_rewards = []
    
    for episode in range(episodes):
        print(f"\n🎯 Episode {episode + 1}/{episodes}")
        
        # Reset environments
        try:
            visual_state, _ = env.reset()
            has_visual = True
        except:
            has_visual = False
            
        agent_state, _ = agent_env.reset()
        agent_state = concat_state_latent(agent_state, skill_id, params["n_skills"])
        
        episode_reward = 0
        step_count = 0
        
        max_steps = min(500, env.spec.max_episode_steps if env.spec.max_episode_steps else 500)
        
        for step in range(max_steps):
            # Get action from agent using processed state
            action = agent.choose_action(agent_state)
            
            # Step environments
            if has_visual:
                try:
                    visual_state, reward, terminated, truncated, _ = env.step(action)
                    env.render()
                except:
                    has_visual = False
                    
            agent_state, reward, agent_terminated, agent_truncated, _ = agent_env.step(action)
            
            done = agent_terminated or agent_truncated
            agent_state = concat_state_latent(agent_state, skill_id, params["n_skills"])
            
            episode_reward += reward
            step_count += 1
            
            if done:
                break
        
        total_rewards.append(episode_reward)
        print(f"   📊 Episode {episode + 1} Reward: {episode_reward}")
        print(f"   ⏱️  Steps: {step_count}")
    
    avg_reward = np.mean(total_rewards)
    print(f"\n🏆 Skill {skill_id} Summary:")
    print(f"   Average Reward: {avg_reward:.1f}")
    print(f"   Individual Rewards: {total_rewards}")
    
    env.close()
    agent_env.close()

if __name__ == "__main__":
    watch_skill() 