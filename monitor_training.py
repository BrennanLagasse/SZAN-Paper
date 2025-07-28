#!/usr/bin/env python3
"""
Simple script to monitor DIAYN training progress
"""
import os
import glob
import time
from pathlib import Path

def check_training_progress():
    """Check the current training progress"""
    
    # Check if training is running
    print("=== DIAYN Training Monitor ===\n")
    
    # Look for checkpoint directories
    checkpoint_dirs = glob.glob("Checkpoints/PongNoFrameskip/*/")
    if checkpoint_dirs:
        latest_dir = max(checkpoint_dirs, key=os.path.getctime)
        print(f"✅ Training session found: {latest_dir}")
        
        # Check for parameter files
        param_files = glob.glob(os.path.join(latest_dir, "*.pth"))
        if param_files:
            latest_checkpoint = max(param_files, key=os.path.getctime)
            print(f"📁 Latest checkpoint: {os.path.basename(latest_checkpoint)}")
        
    else:
        print("❌ No training checkpoints found yet")
    
    # Check log directories
    log_dirs = glob.glob("Logs/PongNoFrameskip/*/")
    if log_dirs:
        latest_log_dir = max(log_dirs, key=os.path.getctime)
        print(f"📊 Log directory: {latest_log_dir}")
    else:
        print("❌ No log directories found yet")
    
    # Check config
    if os.path.exists("config_log.txt"):
        print("\n📋 Current Configuration:")
        with open("config_log.txt", "r") as f:
            for line in f:
                if any(key in line for key in ["n_skills", "max_n_episodes", "max_episode_len", "batch_size"]):
                    print(f"   {line.strip()}")
    
    print("\n" + "="*50)
    print("Training Tips:")
    print("• Training will be slow initially as the replay buffer fills up")
    print("• Each episode takes ~2-3 minutes in early stages")
    print("• Speed will increase as the agent learns")
    print("• Check Logs/ directory for TensorBoard logs")
    print("• Check Checkpoints/ directory for saved models")
    print("="*50)

if __name__ == "__main__":
    check_training_progress() 