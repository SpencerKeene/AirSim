import setup_path
import gym
import airgym
import time

from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback

# Create a DummyVecEnv for main airsim gym env
env = DummyVecEnv(
    [
        lambda: Monitor(
            gym.make(
                "airgym:airsim-drone-sample-v0",
                ip_address="127.0.0.1",
                step_length=0.5,
                image_shape=(84, 84, 1),
            )
        )
    ]
)

# Wrap env as VecTransposeImage to allow SB to handle frame observations
env = VecTransposeImage(env)

# Initialize RL algorithm type and parameters
model = SAC(
    "CnnPolicy",
    env,
    learning_rate=0.00025,
    verbose=1,
    batch_size=32,
    train_freq=4,
    target_update_interval=1000,
    learning_starts=1000,
    buffer_size=500000,
    device="cuda",
    tensorboard_log="./tb_logs/",
)

# Create an evaluation callback with the same env, called every 10000 iterations
callbacks = []
eval_callback = EvalCallback(
    env,
    callback_on_new_best=None,
    n_eval_episodes=5,
    best_model_save_path=".",
    log_path=".",
    eval_freq=500,
)
callbacks.append(eval_callback)

kwargs = {}
kwargs["callback"] = callbacks

train_time = time.time()

# Train for a certain number of timesteps
model.learn(
    total_timesteps=1e5,
    tb_log_name="sac_airsim_drone_run_" + str(train_time),
    **kwargs
)

# Save policy weights
model.save("sac_airsim_drone_policy_" + str(train_time))
