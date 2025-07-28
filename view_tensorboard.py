#!/usr/bin/env python3
"""
Launch TensorBoard to view DIAYN training progress
"""
import subprocess
import os
import glob

def launch_tensorboard():
    # Find the latest log directory
    log_dirs = glob.glob("Logs/PongNoFrameskip/*/")
    
    if not log_dirs:
        print("❌ No training logs found!")
        print("   Make sure you've run training first with:")
        print("   python main.py --do_train --n_skills 10")
        return
    
    # Use the Logs directory as the base
    log_base = "Logs"
    
    print("🚀 Launching TensorBoard...")
    print(f"📊 Log directory: {log_base}")
    print("🌐 Open your browser to: http://localhost:6006")
    print("⏹️  Press Ctrl+C to stop TensorBoard")
    
    try:
        # Launch TensorBoard
        subprocess.run([
            "tensorboard", 
            "--logdir", log_base,
            "--port", "6006",
            "--host", "localhost"
        ])
    except KeyboardInterrupt:
        print("\n✅ TensorBoard stopped.")
    except FileNotFoundError:
        print("❌ TensorBoard not found!")
        print("   Install with: pip install tensorboard")
    except Exception as e:
        print(f"❌ Error launching TensorBoard: {e}")

if __name__ == "__main__":
    launch_tensorboard() 