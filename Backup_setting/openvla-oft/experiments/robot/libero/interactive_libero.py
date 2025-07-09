
"""
run_libero_eval_interactive.py

Interactive version of LIBERO evaluation - allows user to select tasks and input custom commands.
"""

import json
import logging
import os
import sys
from collections import deque
from dataclasses import dataclass
from enum import Enum
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
    get_libero_wrist_image,
    quat2axisangle,
    save_rollout_video,
)
from experiments.robot.openvla_utils import (
    get_action_head,
    get_noisy_action_projector,
    get_processor,
    get_proprio_projector,
    resize_image_for_policy,
)
from experiments.robot.robot_utils import (
    DATE_TIME,
    get_action,
    get_image_resize_size,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)
from prismatic.vla.constants import NUM_ACTIONS_CHUNK


# Define task suite constants
class TaskSuite(str, Enum):
    LIBERO_SPATIAL = "libero_spatial"
    LIBERO_OBJECT = "libero_object"
    LIBERO_GOAL = "libero_goal"
    LIBERO_10 = "libero_10"
    LIBERO_90 = "libero_90"


# Define max steps for each task suite
TASK_MAX_STEPS = {
    TaskSuite.LIBERO_SPATIAL: 220,
    TaskSuite.LIBERO_OBJECT: 280,
    TaskSuite.LIBERO_GOAL: 300,
    TaskSuite.LIBERO_10: 520,
    TaskSuite.LIBERO_90: 400,
}


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


@dataclass
class GenerateConfig:
    # fmt: off

    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "openvla"
    pretrained_checkpoint: Union[str, Path] = ""

    use_l1_regression: bool = True
    use_diffusion: bool = False
    num_diffusion_steps_train: int = 50
    num_diffusion_steps_inference: int = 50
    use_film: bool = False
    num_images_in_input: int = 2
    use_proprio: bool = True

    center_crop: bool = True
    num_open_loop_steps: int = 8

    lora_rank: int = 32

    unnorm_key: Union[str, Path] = ""

    load_in_8bit: bool = False
    load_in_4bit: bool = False

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = TaskSuite.LIBERO_10  # Default to LIBERO_10 for interactive mode
    num_steps_wait: int = 10
    initial_states_path: str = "DEFAULT"
    env_img_res: int = 256

    #################################################################################################################
    # Interactive mode parameters
    #################################################################################################################
    interactive_mode: bool = True  # Enable interactive mode
    num_episodes_per_command: int = 5  # Number of episodes to run per command

    #################################################################################################################
    # Utils
    #################################################################################################################
    run_id_note: Optional[str] = None
    local_log_dir: str = "./experiments/logs"

    use_wandb: bool = False
    wandb_entity: str = "your-wandb-entity"
    wandb_project: str = "your-wandb-project"

    seed: int = 7

    # fmt: on


def validate_config(cfg: GenerateConfig) -> None:
    """Validate configuration parameters."""
    assert cfg.pretrained_checkpoint is not None, "pretrained_checkpoint must not be None!"

    if "image_aug" in str(cfg.pretrained_checkpoint):
        assert cfg.center_crop, "Expecting `center_crop==True` because model was trained with image augmentations!"

    assert not (cfg.load_in_8bit and cfg.load_in_4bit), "Cannot use both 8-bit and 4-bit quantization!"

    # Validate task suite
    assert cfg.task_suite_name in [suite.value for suite in TaskSuite], f"Invalid task suite: {cfg.task_suite_name}"


def initialize_model(cfg: GenerateConfig):
    """Initialize model and associated components."""
    # Load model
    model = get_model(cfg)

    # Load proprio projector if needed
    proprio_projector = None
    if cfg.use_proprio:
        proprio_projector = get_proprio_projector(
            cfg,
            model.llm_dim,
            proprio_dim=8,  # 8-dimensional proprio for LIBERO
        )

    # Load action head if needed
    action_head = None
    if cfg.use_l1_regression or cfg.use_diffusion:
        action_head = get_action_head(cfg, model.llm_dim)

    # Load noisy action projector if using diffusion
    noisy_action_projector = None
    if cfg.use_diffusion:
        noisy_action_projector = get_noisy_action_projector(cfg, model.llm_dim)

    # Get OpenVLA processor if needed
    processor = None
    if cfg.model_family == "openvla":
        processor = get_processor(cfg)
        check_unnorm_key(cfg, model)

    return model, action_head, proprio_projector, noisy_action_projector, processor


def check_unnorm_key(cfg: GenerateConfig, model) -> None:
    """Check that the model contains the action un-normalization key."""
    # Initialize unnorm_key
    unnorm_key = cfg.task_suite_name

    # In some cases, the key must be manually modified
    if unnorm_key not in model.norm_stats and f"{unnorm_key}_no_noops" in model.norm_stats:
        unnorm_key = f"{unnorm_key}_no_noops"

    assert unnorm_key in model.norm_stats, f"Action un-norm key {unnorm_key} not found in VLA `norm_stats`!"

    # Set the unnorm_key in cfg
    cfg.unnorm_key = unnorm_key


def setup_logging(cfg: GenerateConfig):
    """Set up logging to file and optionally to wandb."""
    # Create run ID
    run_id = f"EVAL-INTERACTIVE-{cfg.task_suite_name}-{cfg.model_family}-{DATE_TIME}"
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"

    # Set up local logging
    os.makedirs(cfg.local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(cfg.local_log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    logger.info(f"Logging to local log file: {local_log_filepath}")

    # Initialize Weights & Biases logging if enabled
    if cfg.use_wandb:
        wandb.init(
            entity=cfg.wandb_entity,
            project=cfg.wandb_project,
            name=run_id,
        )

    return log_file, local_log_filepath, run_id


def log_message(message: str, log_file=None):
    """Log a message to console and optionally to a log file."""
    logger.info(message)
    if log_file:
        log_file.write(message + "\n")
        log_file.flush()


def load_initial_states(cfg: GenerateConfig, task_suite, task_id: int, log_file=None):
    """Load initial states for the given task."""
    # Get default initial states
    initial_states = task_suite.get_task_init_states(task_id)

    # If using custom initial states, load them from file
    if cfg.initial_states_path != "DEFAULT":
        with open(cfg.initial_states_path, "r") as f:
            all_initial_states = json.load(f)
        log_message(f"Using initial states from {cfg.initial_states_path}", log_file)
        return initial_states, all_initial_states
    else:
        log_message("Using default initial states", log_file)
        return initial_states, None


def prepare_observation(obs, resize_size):
    """Prepare observation for policy input."""
    # Get preprocessed images
    img = get_libero_image(obs)
    wrist_img = get_libero_wrist_image(obs)

    # Resize images to size expected by model
    img_resized = resize_image_for_policy(img, resize_size)
    wrist_img_resized = resize_image_for_policy(wrist_img, resize_size)

    # Prepare observations dict
    observation = {
        "full_image": img_resized,
        "wrist_image": wrist_img_resized,
        "state": np.concatenate(
            (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
        ),
    }

    return observation, img


def process_action(action, model_family):
    """Process action before sending to environment."""
    # Normalize gripper action [0,1] -> [-1,+1] because the environment expects the latter
    action = normalize_gripper_action(action, binarize=True)

    # [OpenVLA] The dataloader flips the sign of the gripper action
    if model_family == "openvla":
        action = invert_gripper_action(action)

    return action


def run_episode(
    cfg: GenerateConfig,
    env,
    task_description: str,
    model,
    resize_size,
    processor=None,
    action_head=None,
    proprio_projector=None,
    noisy_action_projector=None,
    initial_state=None,
    log_file=None,
):
    """Run a single episode in the environment."""
    # Reset environment
    env.reset()

    # Set initial state if provided
    if initial_state is not None:
        obs = env.set_init_state(initial_state)
    else:
        obs = env.get_observation()

    # Initialize action queue
    if cfg.num_open_loop_steps != NUM_ACTIONS_CHUNK:
        print(f"WARNING: cfg.num_open_loop_steps ({cfg.num_open_loop_steps}) does not match the NUM_ACTIONS_CHUNK "
              f"({NUM_ACTIONS_CHUNK}) constant!")
    action_queue = deque(maxlen=cfg.num_open_loop_steps)

    # Setup
    t = 0
    replay_images = []
    max_steps = TASK_MAX_STEPS[cfg.task_suite_name]

    # Run episode
    success = False
    try:
        while t < max_steps + cfg.num_steps_wait:
            # Do nothing for the first few timesteps to let objects stabilize
            if t < cfg.num_steps_wait:
                obs, reward, done, info = env.step(get_libero_dummy_action(cfg.model_family))
                t += 1
                continue

            # Prepare observation
            observation, img = prepare_observation(obs, resize_size)
            replay_images.append(img)

            # If action queue is empty, requery model
            if len(action_queue) == 0:
                # Query model to get action
                actions = get_action(
                    cfg,
                    model,
                    observation,
                    task_description,
                    processor=processor,
                    action_head=action_head,
                    proprio_projector=proprio_projector,
                    noisy_action_projector=noisy_action_projector,
                    use_film=cfg.use_film,
                )
                action_queue.extend(actions)

            # Get action from queue
            action = action_queue.popleft()

            # Process action
            action = process_action(action, cfg.model_family)

            # Execute action in environment
            obs, reward, done, info = env.step(action.tolist())
            if done:
                success = True
                break
            t += 1

    except Exception as e:
        log_message(f"Episode error: {e}", log_file)

    return success, replay_images


def display_task_list(task_suite):
    """Display available tasks in the task suite."""
    print("\n" + "="*60)
    print("AVAILABLE TASKS:")
    print("="*60)
    for task_id in range(task_suite.n_tasks):
        task = task_suite.get_task(task_id)
        _, task_description = get_libero_env(task, "openvla", resolution=256)
        print(f"{task_id + 1:2d}. {task_description}")
    print("="*60)


def get_user_task_selection(task_suite):
    """Get task selection from user."""
    while True:
        try:
            choice = input(f"\nSelect task (1-{task_suite.n_tasks}) or 'q' to quit: ").strip()
            if choice.lower() == 'q':
                return None
            task_id = int(choice) - 1
            if 0 <= task_id < task_suite.n_tasks:
                return task_id
            else:
                print(f"Please enter a number between 1 and {task_suite.n_tasks}")
        except ValueError:
            print("Please enter a valid number or 'q' to quit")


def get_user_command():
    """Get custom command from user."""
    while True:
        command = input("\nEnter custom command (or 'back' to return to task selection): ").strip()
        if command.lower() == 'back':
            return None
        if command:
            return command
        print("Please enter a valid command")


def run_interactive_episodes(
    cfg: GenerateConfig,
    task_suite,
    task_id: int,
    custom_command: str,
    model,
    resize_size,
    processor=None,
    action_head=None,
    proprio_projector=None,
    noisy_action_projector=None,
    log_file=None,
):
    """Run episodes with custom command for selected task."""
    # Get task and initial states
    task = task_suite.get_task(task_id)
    initial_states, all_initial_states = load_initial_states(cfg, task_suite, task_id, log_file)

    # Initialize environment
    env, original_task_description = get_libero_env(task, cfg.model_family, resolution=cfg.env_img_res)
    
    # Use custom command instead of original task description
    log_message(f"Original task: {original_task_description}", log_file)
    log_message(f"Custom command: {custom_command}", log_file)

    # Run episodes
    successes = 0
    for episode_idx in range(cfg.num_episodes_per_command):
        log_message(f"\nEpisode {episode_idx + 1}/{cfg.num_episodes_per_command}", log_file)
        log_message(f"Task: {custom_command}", log_file)

        # Handle initial state
        if cfg.initial_states_path == "DEFAULT":
            # Use default initial state (cycling through available states)
            initial_state = initial_states[episode_idx % len(initial_states)]
        else:
            # Get keys for fetching initial episode state from JSON
            initial_states_task_key = original_task_description.replace(" ", "_")
            episode_key = f"demo_{episode_idx % len(initial_states)}"

            # Skip episode if expert demonstration failed
            if not all_initial_states[initial_states_task_key][episode_key]["success"]:
                log_message(f"Skipping episode {episode_idx} due to failed expert demo!", log_file)
                continue

            # Get initial state
            initial_state = np.array(all_initial_states[initial_states_task_key][episode_key]["initial_state"])

        # Run episode with custom command
        success, replay_images = run_episode(
            cfg,
            env,
            custom_command,  # Use custom command here
            model,
            resize_size,
            processor,
            action_head,
            proprio_projector,
            noisy_action_projector,
            initial_state,
            log_file,
        )

        if success:
            successes += 1

        # Save replay video
        save_rollout_video(
            replay_images, 
            episode_idx + 1, 
            success=success, 
            task_description=f"{custom_command} (Task {task_id + 1})", 
            log_file=log_file
        )

        # Log episode result
        log_message(f"Episode {episode_idx + 1} Success: {success}", log_file)

    # Calculate and log success rate
    success_rate = successes / cfg.num_episodes_per_command
    log_message(f"\nResults for '{custom_command}':", log_file)
    log_message(f"Success rate: {success_rate:.2f} ({successes}/{cfg.num_episodes_per_command})", log_file)

    # Log to wandb if enabled
    if cfg.use_wandb:
        wandb.log({
            f"success_rate/{custom_command}": success_rate,
            f"num_episodes/{custom_command}": cfg.num_episodes_per_command,
        })

    return success_rate


@draccus.wrap()
def eval_libero_interactive(cfg: GenerateConfig) -> None:
    """Interactive LIBERO evaluation - allows user to select tasks and input custom commands."""
    # Validate configuration
    validate_config(cfg)

    # Set random seed
    set_seed_everywhere(cfg.seed)

    # Initialize model and components
    model, action_head, proprio_projector, noisy_action_projector, processor = initialize_model(cfg)

    # Get expected image dimensions
    resize_size = get_image_resize_size(cfg)

    # Setup logging
    log_file, local_log_filepath, run_id = setup_logging(cfg)

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()

    log_message(f"Task suite: {cfg.task_suite_name}", log_file)
    log_message(f"Interactive mode enabled", log_file)
    log_message(f"Episodes per command: {cfg.num_episodes_per_command}", log_file)

    print(f"\n> Interactive LIBERO Evaluation Started!")
    print(f"Task Suite: {cfg.task_suite_name}")
    print(f"Episodes per command: {cfg.num_episodes_per_command}")

    # Main interactive loop
    while True:
        # Display available tasks
        display_task_list(task_suite)

        # Get task selection
        task_id = get_user_task_selection(task_suite)
        if task_id is None:
            break

        # Get original task description for reference
        task = task_suite.get_task(task_id)
        _, original_task_description = get_libero_env(task, cfg.model_family, resolution=cfg.env_img_res)
        
        print(f"\nSelected Task {task_id + 1}: {original_task_description}")

        # Inner loop for custom commands on selected task
        while True:
            custom_command = get_user_command()
            if custom_command is None:
                break

            print(f"\nRunning {cfg.num_episodes_per_command} episodes with command: '{custom_command}'")
            
            # Run episodes with custom command
            success_rate = run_interactive_episodes(
                cfg,
                task_suite,
                task_id,
                custom_command,
                model,
                resize_size,
                processor,
                action_head,
                proprio_projector,
                noisy_action_projector,
                log_file,
            )

            print(f" Completed! Success rate: {success_rate:.2f}")

    # Close log file
    log_message("Interactive evaluation ended", log_file)
    if log_file:
        log_file.close()

    # Save wandb log if enabled
    if cfg.use_wandb:
        wandb.save(local_log_filepath)

    print("\n=K Interactive evaluation completed!")


if __name__ == "__main__":
    eval_libero_interactive() 