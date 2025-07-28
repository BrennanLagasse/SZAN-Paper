#!/usr/bin/env python3
"""
Analyze the learned DIAYN skills without video generation
"""
import gymnasium as gym
import ale_py
import numpy as np
from Brain import SACAgent
from Common import Logger, get_params, make_atari_env

def concat_state_latent(s, z_, n):
    z_one_hot = np.zeros(n)
    z_one_hot[z_] = 1
    return np.concatenate([s, z_one_hot])

def analyze_skills():
    """Test each skill and see how they perform"""
    
    # Load parameters
    params = get_params()
    gym.register_envs(ale_py)
    
    # Create environment
    env = make_atari_env(params["env_name"])
    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.n
    action_bounds = [0, env.action_space.n - 1]

    params.update({"n_states": n_states,
                   "n_actions": n_actions,
                   "action_bounds": action_bounds})

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
    
    print(f"\n🎮 Testing {params['n_skills']} learned skills on Pong...")
    print("="*60)
    
    skill_performances = {}
    
    for skill_id in range(params["n_skills"]):
        print(f"\n🤖 Testing Skill {skill_id}...")
        
        # Test this skill for a few episodes
        episode_rewards = []
        episode_lengths = []
        
        for episode in range(3):  # Test 3 episodes per skill
            state, _ = env.reset()
            state = concat_state_latent(state, skill_id, params["n_skills"])
            episode_reward = 0
            episode_length = 0
            
            max_steps = min(500, env.spec.max_episode_steps if env.spec.max_episode_steps else 500)
            
            for step in range(max_steps):
                action = agent.choose_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                next_state = concat_state_latent(next_state, skill_id, params["n_skills"])
                
                episode_reward += reward
                episode_length += 1
                state = next_state
                
                if done:
                    break
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
        
        # Calculate statistics for this skill
        avg_reward = np.mean(episode_rewards)
        avg_length = np.mean(episode_lengths)
        
        skill_performances[skill_id] = {
            'avg_reward': avg_reward,
            'avg_length': avg_length,
            'rewards': episode_rewards
        }
        
        print(f"   📊 Average Reward: {avg_reward:.1f}")
        print(f"   ⏱️  Average Length: {avg_length:.1f} steps")
        print(f"   📈 Individual rewards: {episode_rewards}")
    
    # Summary
    print("\n" + "="*60)
    print("🏆 SKILL ANALYSIS SUMMARY")
    print("="*60)
    
    # Sort skills by performance
    sorted_skills = sorted(skill_performances.items(), 
                          key=lambda x: x[1]['avg_reward'], 
                          reverse=True)
    
    for rank, (skill_id, perf) in enumerate(sorted_skills, 1):
        print(f"{rank:2d}. Skill {skill_id}: "
              f"Reward={perf['avg_reward']:6.1f}, "
              f"Length={perf['avg_length']:6.1f}")
    
    # Diversity analysis
    rewards = [perf['avg_reward'] for perf in skill_performances.values()]
    lengths = [perf['avg_length'] for perf in skill_performances.values()]
    
    print(f"\n📊 DIVERSITY METRICS:")
    print(f"   Reward std dev: {np.std(rewards):.2f}")
    print(f"   Length std dev: {np.std(lengths):.2f}")
    print(f"   Best skill reward: {max(rewards):.1f}")
    print(f"   Worst skill reward: {min(rewards):.1f}")
    print(f"   Range: {max(rewards) - min(rewards):.1f}")
    
    env.close()
    
    return skill_performances

if __name__ == "__main__":
    analyze_skills() 