#!/usr/bin/env python
import sys
import os
import wandb
import socket
import setproctitle
import numpy as np
from pathlib import Path
import torch
from onpolicy.config import get_config
from onpolicy.envs.pong.Pong_env import PongEnv
from onpolicy.envs.pong.VMAPD_wrapper import VMAPDWrapper
from onpolicy.envs.env_wrappers import SubprocVecEnv, DummyVecEnv

"""Train script for Pong."""

def make_train_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "pong":
                env = PongEnv(all_args)
                # env = VMAPDWrapper(env, all_args.max_z, rank%all_args.max_z)   #old
                env = VMAPDWrapper(env, all_args.max_z, None)  # None instead of rank%all_args.max_z that fixed only picking skill 1 problem
            else:
                print("Can not support the " +
                      all_args.env_name + "environment.")
                raise NotImplementedError
            env.seed(all_args.seed + rank * 1000)
            return env
        return init_env
    if all_args.n_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])


def make_eval_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "pong":
                env = PongEnv(all_args)
                env = VMAPDWrapper(env, all_args.max_z, rank%all_args.max_z)
            else:
                print("Can not support the " +
                      all_args.env_name + "environment.")
                raise NotImplementedError
            env.seed(all_args.seed * 50000 + rank * 10000)
            return env
        return init_env
    if all_args.n_eval_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)])


def parse_args(args, parser):
    parser.add_argument('--game_name', type=str, default='PongNoFrameskip-v4', help="Which Atari game to run on")
    parser.add_argument('--num_agents', type=int, default=1, help="number of players")  # Pong is single agent
    all_args = parser.parse_known_args(args)[0]
    return all_args


def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)

    assert (all_args.use_recurrent_policy or all_args.use_naive_recurrent_policy), ("check recurrent policy!")  

    # cuda
    if all_args.cuda and torch.cuda.is_available():
        print("choose to use gpu...")
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("choose to use cpu...")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    # run dir
    run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0] + "/results") \
         / all_args.env_name / all_args.game_name / all_args.algorithm_name / all_args.experiment_name
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    # wandb
    if all_args.use_wandb:
        run = wandb.init(config=all_args,
                        #  project="VMAPD",
                         project="pong-training",
                        #  entity=all_args.user_name,
                         entity="ncataldo01278",
                         notes=socket.gethostname(),
                         name=str(all_args.algorithm_name) + "_" +
                         str(all_args.game_name) +
                         "_seed" + str(all_args.seed),
                         group=all_args.game_name,
                         dir=str(run_dir),
                         job_type="training",
                         reinit=True)
    else:
        if not run_dir.exists():
            curr_run = 'run1'
        else:
            exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in run_dir.iterdir() if str(folder.name).startswith('run')]
            if len(exst_run_nums) == 0:
                curr_run = 'run1'
            else:
                curr_run = 'run%i' % (max(exst_run_nums) + 1)
        run_dir = run_dir / curr_run
        if not run_dir.exists():
            os.makedirs(str(run_dir))

    setproctitle.setproctitle(str(all_args.algorithm_name) + "-" + \
        str(all_args.env_name) + "-" + str(all_args.experiment_name) + "@" + str(all_args.user_name))

    # seed
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    # env init
    envs = make_train_env(all_args)
    eval_envs = make_eval_env(all_args) if all_args.use_eval else None
    num_agents = all_args.num_agents

    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": num_agents,
        "device": device,
        "run_dir": run_dir
    }

    # run experiments
    if all_args.share_policy:
        from onpolicy.runner.shared.pong_runner import PongRunner as Runner
    else:
        raise NotImplementedError

    runner = Runner(config)
    runner.run()
    
    # post process
    envs.close()
    if all_args.use_eval and eval_envs is not envs:
        eval_envs.close()

    if all_args.use_wandb:
        run.finish()
    else:
        runner.writter.export_scalars_to_json(str(runner.log_dir + '/summary.json'))
        runner.writter.close()






if __name__ == "__main__":
    import sys
    test_args = [
        'train_atari.py',
        '--env_name', 'pong',
        '--game_name', 'PongNoFrameskip-v4',
        '--algorithm_name', 'ours',      
        '--max_z', '2',
        '--div_thresh', '0.8',
        '--rex_thresh', '0.8',
        '--alpha_rex', '1.0',
        '--alpha_div', '1.0',
        '--n_rollout_threads', '8', #shoudl be 16
        '--episode_length', '128',    #shoudl be 128
        '--ppo_epoch', '4',
        '--num_mini_batch', '4',
        '--num_env_steps', '50000000',  
        '--experiment_name', 'Pong_Training_8_29Test',
        '--seed', '42',
        '--use_wandb', 'False'
    ]
    main(test_args)






