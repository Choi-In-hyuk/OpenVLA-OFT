"""
run_libero_eval_interactive.py

Runs a model in a LIBERO simulation environment with interactive task selection and custom commands.

Usage:
    python experiments/robot/libero/run_libero_eval_interactive.py \
        --model_family openvla \
        --pretrained_checkpoint <CHECKPOINT_PATH> \
        --task_suite_name [ libero_spatial | libero_object | libero_goal | libero_10 | libero_90 ] \
        --center_crop [ True | False ] \
        --run_id_note <OPTIONAL TAG TO INSERT INTO RUN ID FOR LOGGING> \
        --use_wandb [ True | False ] \
        --wandb_project <PROJECT> \
        --wandb_entity <ENTITY>
"""

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import draccus
import numpy as np
import tqdm
from libero.libero import benchmark

import wandb

# Append current directory so that interpreter can find experiments.robot
sys.path.append("../..")
from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    quat2axisangle,
    save_rollout_video,
)
from experiments.robot.openvla_utils import get_processor
from experiments.robot.robot_utils import (
    DATE_TIME,
    get_action,
    get_image_resize_size,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)


@dataclass
class GenerateConfig:
    # fmt: off

    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "openvla"                    # Model family
    pretrained_checkpoint: Union[str, Path] = ""     # Pretrained checkpoint path
    load_in_8bit: bool = False                       # (For OpenVLA only) Load with 8-bit quantization
    load_in_4bit: bool = False                       # (For OpenVLA only) Load with 4-bit quantization

    center_crop: bool = True                         # Center crop? (if trained w/ random crack image aug)

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = "libero_spatial"          # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    num_steps_wait: int = 10                         # Number of steps to wait for objects to stabilize in sim
    
    #################################################################################################################
    # Interactive mode parameters
    #################################################################################################################
    episode_num_for_each_task: int = 1               # Number of episodes to run for each selected task
    max_episode_length: int = 300                    # Maximum episode length (will be adjusted per task suite)

    #################################################################################################################
    # Utils
    #################################################################################################################
    run_id_note: Optional[str] = None                # Extra note to add in run ID for logging
    local_log_dir: str = "./experiments/logs"        # Local directory for eval logs

    use_wandb: bool = False                          # Whether to also log results in Weights & Biases
    wandb_project: str = "YOUR_WANDB_PROJECT"        # Name of W&B project to log to (use default!)
    wandb_entity: str = "YOUR_WANDB_ENTITY"          # Name of entity to log under

    seed: int = 7                                    # Random Seed (for reproducibility)

    # fmt: on


def print_task_list(task_suite, task_suite_name):
    """Print available tasks in the suite"""
    print(f"\n=== Available Tasks in {task_suite_name} ===")
    for i in range(task_suite.n_tasks):
        task = task_suite.get_task(i)
        print(f"{i+1}. {task.problem_info}")
    print("=" * 50)


def get_user_task_selection(task_suite):
    """Get user's task selection"""
    while True:
        try:
            selection = input(f"\nSelect a task (1-{task_suite.n_tasks}) or 'q' to quit: ").strip()
            if selection.lower() == 'q':
                return None
            task_id = int(selection) - 1
            if 0 <= task_id < task_suite.n_tasks:
                return task_id
            else:
                print(f"Please enter a number between 1 and {task_suite.n_racks}")
        except ValueError:
            print("Please enter a valid number or 'q' to quit.")


def get_custom_command():
    """Get custom command from user"""
    print("\nEnter your custom command for the robot:")
    print("(Press Enter to use the original task description)")
    command = input("Command: ").strip()
    return command if command else None


def run_single_episode(cfg, model, processor, env, task_description, episode_num, log_file):
    """Run a single episode"""
    print(f"\n=== Starting Episode {episode_num} ===")
    print(f"Task: {task_description}")
    log_file.write(f"\nEpisode {episode_num} - Task: {task_description}\n")
    
    # Reset environment
    env.reset()
    
    # Set initial states (use the first initial state for simplicity)
    obs = env.set_init_state(env.get_task_init_states(0)[0])
    
    # Setup
    t = 0
    replay_images = []
    
    # Set max steps based on task suite
    if cfg.task_suite_name == "libero_spatial":
        max_steps = 220
    elif cfg.task_suite_name == "libero_object":
        max_steps = 280
    elif cfg.task_suite_name == "libero_goal":
        max_steps = 300
    elif cfg.task_suite_name == "libero_10":
        max_steps = 520
    elif cfg.task_suite_name == "libero_90":
        max_steps = 400
    else:
        max_steps = cfg.max_episode_length
    
    success = False
    
    while t < max_steps + cfg.num_steps_wait:
        try:
            # Wait for objects to stabilize
            if t < cfg.num_steps_wait:
                obs, reward, done, info = env.step(get_libero_dummy_action(cfg.model_family))
                t += 1
                continue
            
            # Get preprocessed image
            resize_size = get_image_resize_size(cfg)
            img = get_libero_image(obs, resize_size)
            
            # Save preprocessed image for replay video
            replay_images.append(img)
            
            # Prepare observations dict
            observation = {
                "full_image": img,
                "state": np.concatenate(
                    (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
                ),
            }
            
            # Query model to get action
            action = get_action(
                cfg,
                model,
                observation,
                task_description,
                processor=processor,
            )
            
            # Normalize gripper action [0,1] -> [-1,+1]
            action = normalize_gripper_action(action, binarize=True)
            
            # Invert gripper action for OpenVLA
            if cfg.model_family == "openvla":
                action = invert_gripper_action(action)
            
            # Execute action in environment
            obs, reward, done, info = env.step(action.tolist())
            
            if done:
                success = True
                break
            
            t += 1
            
            # Print progress every 50 steps
            if t % 50 == 0:
                print(f"Step {t}/{max_steps + cfg.num_steps_wait}")
                
        except Exception as e:
            print(f"Caught exception: {e}")
            log_file.write(f"Caught exception: {e}\n")
            break
    
    # Save replay video
    save_rollout_video(
        replay_images, episode_num, success=success, task_description=task_description, log_file=log_file
    )
    
    # Log results
    result_msg = f"Episode {episode_num} completed. Success: {success} (Steps: {t})"
    print(result_msg)
    log_file.write(result_msg + "\n")
    log_file.flush()
    
    return success


@draccus.wrap()
def eval_libero(cfg: GenerateConfig) -> None:
    assert cfg.pretrained_checkpoint is not None, "cfg.pretrained_checkpoint must not be None!"
    if "image_aug" in cfg.pretrained_checkpoint:
        assert cfg.center_crop, "Expecting `center_crop==True` because model was trained with image augmentations!"
    assert not (cfg.load_in_8bit and cfg.load_in_4bit), "Cannot use both 8-bit and 4-bit quantization!"

    # Set random seed
    set_seed_everywhere(cfg.seed)

    # Set action un-normalization key
    cfg.unnorm_key = cfg.task_suite_name

    # Load model
    print("Loading model...")
    model = get_model(cfg)

    # Check action un-normalization key for OpenVLA
    if cfg.model_family == "openvla":
        if cfg.unnorm_key not in model.norm_stats and f"{cfg.unnorm_key}_no_noops" in model.norm_stats:
            cfg.unnorm_key = f"{cfg.unnorm_key}_no_noops"
        assert cfg.unnorm_key in model.norm_stats, f"Action un-norm key {cfg.unnorm_key} not found in VLA `norm_stats`!"

    # Get processor for OpenVLA
    processor = None
    if cfg.model_family == "openvla":
        processor = get_processor(cfg)

    # Initialize logging
    run_id = f"INTERACTIVE-{cfg.task_suite_name}-{cfg.model_family}-{DATE_TIME}"
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"
    os.makedirs(cfg.local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(cfg.local_log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    print(f"Logging to: {local_log_filepath}")

    # Initialize Weights & Biases
    if cfg.use_wandb:
        wandb.init(
            entity=cfg.wandb_entity,
            project=cfg.wandb_project,
            name=run_id,
        )

    # Initialize LIBERO task suite
    print("Loading LIBERO task suite...")
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    
    print(f"Loaded task suite: {cfg.task_suite_name}")
    log_file.write(f"Task suite: {cfg.task_suite_name}\n")
    log_file.write(f"Model: {cfg.model_family}\n")
    log_file.write(f"Checkpoint: {cfg.pretrained_checkpoint}\n\n")
    
    # Interactive loop
    total_episodes = 0
    total_successes = 0
    
    while True:
        # Show available tasks
        print_task_list(task_suite, cfg.task_suite_name)
        
        # Get user task selection
        task_id = get_user_task_selection(task_suite)
        if task_id is None:
            break
        
        # Get task and its original description
        task = task_suite.get_task(task_id)
        original_task_description = task.problem_info
        
        # Get custom command from user
        custom_command = get_custom_command()
        task_description = custom_command if custom_command else original_task_description
        
        print(f"\nSelected task: {original_task_description}")
        if custom_command:
            print(f"Custom command: {custom_command}")
        
        # Ask for number of episodes
        while True:
            try:
                num_episodes = input(f"Number of episodes to run (default: {cfg.episode_num_for_each_task}): ").strip()
                if not num_episodes:
                    num_episodes = cfg.episode_num_for_each_task
                else:
                    num_episodes = int(num_episodes)
                if num_episodes > 0:
                    break
                else:
                    print("Please enter a positive number.")
            except ValueError:
                print("Please enter a valid number.")
        
        # Initialize environment
        env, _ = get_libero_env(task, cfg.model_family, resolution=256)
        
        # Run episodes
        task_successes = 0
        for episode_idx in range(num_episodes):
            success = run_single_episode(
                cfg, model, processor, env, task_description, episode_idx + 1, log_file
            )
            if success:
                task_successes += 1
                total_successes += 1
            total_episodes += 1
        
        # Log results for this task
        task_success_rate = task_successes / num_episodes
        overall_success_rate = total_successes / total_episodes
        
        print(f"\n=== Task Results ===")
        print(f"Task: {original_task_description}")
        if custom_command:
            print(f"Custom command: {custom_command}")
        print(f"Episodes: {num_episodes}")
        print(f"Successes: {task_successes}")
        print(f"Success rate: {task_success_rate:.2%}")
        print(f"Overall success rate: {overall_success_rate:.2%} ({total_successes}/{total_episodes})")
        
        log_file.write(f"\n=== Task Results ===\n")
        log_file.write(f"Task: {original_task_description}\n")
        if custom_command:
            log_file.write(f"Custom command: {custom_command}\n")
        log_file.write(f"Episodes: {num_episodes}\n")
        log_file.write(f"Successes: {task_successes}\n")
        log_file.write(f"Success rate: {task_success_rate:.2%}\n")
        log_file.write(f"Overall success rate: {overall_success_rate:.2%} ({total_successes}/{total_episodes})\n")
        log_file.flush()
        
        # Log to wandb
        if cfg.use_wandb:
            wandb.log({
                f"task_success_rate/{original_task_description}": task_success_rate,
                f"episodes/{original_task_description}": num_episodes,
                "overall_success_rate": overall_success_rate,
                "total_episodes": total_episodes,
                "total_successes": total_successes,
            })
        
        # Ask if user wants to continue
        continue_choice = input("\nWould you like to try another task? (y/n): ").strip().lower()
        if continue_choice not in ['y', 'yes']:
            break
    
    # Final results
    print(f"\n=== Final Results ===")
    print(f"Total episodes: {total_episodes}")
    print(f"Total successes: {total_successes}")
    print(f"Overall success rate: {total_successes/total_episodes:.2%}" if total_episodes > 0 else "No episodes run")
    
    log_file.write(f"\n=== Final Results ===\n")
    log_file.write(f"Total episodes: {total_episodes}\n")
    log_file.write(f"Total successes: {total_successes}\n")
    log_file.write(f"Overall success rate: {total_successes/total_episodes:.2%}\n" if total_episodes > 0 else "No episodes run\n")
    
    # Close log file
    log_file.close()
    
    # Save to wandb
    if cfg.use_wandb:
        if total_episodes > 0:
            wandb.log({
                "final_success_rate": total_successes / total_episodes,
                "final_total_episodes": total_episodes,
                "final_total_successes": total_successes,
            })
        wandb.save(local_log_filepath)
    
    print(f"\nEvaluation completed. Results saved to: {local_log_filepath}")


if __name__ == "__main__":
    eval_libero()