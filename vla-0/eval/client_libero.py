# client_libero.py
import collections
import dataclasses
import logging
import pathlib
import pandas as pd

import imageio
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
import tqdm
import tyro
from PIL import Image

from client_policy import VLA0ClientPolicy
from image_tools import resize_with_pad, convert_to_uint8, _quat2axisangle

# get the current date for logging
from datetime import datetime
current_date = datetime.now().strftime("%Y-%m-%d")

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 224

@dataclasses.dataclass
class Args:
    # Arguments for connecting to the VLA-0 policy server
    host: str = "localhost" # Suggested to use "localhost" when server is on the same machine
    port: int = 8000
    resize_size: int = 224 # Keep consistent with training image resolution

    # LIBERO environment parameters
    task_suite_name: str = (
         "libero_spatial"  # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
        # "libero_object"  # Medium difficulty tasks
        # "libero_goal"  # More difficult tasks
        # "libero_10"  # Highly difficult tasks
    )
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize in sim

    num_trials_per_task: int = 10

    # Video output path
    video_out_path: str = f"data/{current_date}/vla0_eval_videos"
    seed: int = 42

def eval_libero(args: Args) -> None:
    np.random.seed(args.seed)
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    logging.info(f"Task suite: {args.task_suite_name}")
    
    # Make sure the video output directory exists
    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    if args.task_suite_name == "libero_spatial":
        max_steps = 220  # longest training demo has 193 steps
    elif args.task_suite_name == "libero_object":
        max_steps = 280  # longest training demo has 254 steps
    elif args.task_suite_name == "libero_goal":
        max_steps = 300  # longest training demo has 270 steps
    elif args.task_suite_name == "libero_10":
        max_steps = 520  # longest training demo has 505 steps
    elif args.task_suite_name == "libero_90":
        max_steps = 400  # longest training demo has 373 steps
    else:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")

    # Connect to the VLA-0 policy server
    client = VLA0ClientPolicy(args.host, args.port)

    total_episodes, total_successes = 0, 0
    success_rate_list_per_task: list = []  # List to store success rates of each task

    for task_id in tqdm.tqdm(range(task_suite.n_tasks), desc="Tasks"):
        # Get task and its details
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        env, task_description = get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        # Start episodes
        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task), desc="Trials", leave=False):
            logging.info(f"\nStarting task: {task.language}")

            # Reset environment
            env.reset()

            # Set initial states
            obs = env.set_init_state(initial_states[episode_idx])

            replay_images = []
            
            for t in range(max_steps + args.num_steps_wait):

                # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                # and we need to wait for them to fall
                if t < args.num_steps_wait:
                    obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                    t += 1
                    continue

                # Image preprocessing
                main_img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])

                # Use padded resize
                main_img_processed = convert_to_uint8(
                    resize_with_pad(main_img, args.resize_size, args.resize_size)
                )

                wrist_img_processed = convert_to_uint8(
                    resize_with_pad(wrist_img, args.resize_size, args.resize_size)
                )

                replay_images.append(main_img)

                # Prepare observation data to send to the server
                obs_dict = {
                    "observation/image": main_img_processed,
                    "observation/wrist_image": wrist_img_processed,
                    "prompt": str(task.language),
                }

                # get action from the server
                action = client.infer(obs_dict)["actions"]
                obs, reward, done, info = env.step(action)
                
                if info["success"]:
                    total_episodes += 1
                    task_successes += 1
                    break

            # Save video of the episode
            video_name = f"{args.task_suite_name}_task{task_id}_trial{episode_idx}_{'success' if info['success'] else 'fail'}.mp4"
            imageio.mimwrite(pathlib.Path(args.video_out_path) / video_name, replay_images, fps=10)
            
        total_successes += task_successes
        total_episodes += args.num_trials_per_task
        success_rate_list_per_task.append(task_successes / task_episodes)
        logging.info(f"Success rate for task {task_id}: {task_successes / task_episodes}")
        logging.info(f"Overall success rate: {total_successes / total_episodes:.2f}")

    logging.info(f"\nEvaluation complete! Overall success rate: {total_successes / total_episodes:.2f} ({total_successes}/{total_episodes})")
    logging.info(f"Total number of episodes: {total_episodes}")
    logging.info(f"Total number of successes: {total_successes}")
    logging.info(f"Total number of failures: {total_episodes - total_successes}")

    # save the detailed evaluation results to a csv file
    df = pd.DataFrame({
        "task_id": [task_id] * len(success_rate_list_per_task),
        "success_rate": success_rate_list_per_task
    })
    df.to_csv(pathlib.Path(args.video_out_path) / f"{args.task_suite_name}_task{task_id}_evaluation.csv", index=False)

def get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    tyro.cli(eval_libero)