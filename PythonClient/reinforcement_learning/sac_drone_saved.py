import setup_path
import gym
import airgym
import time
import sys

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
                step_length=1,
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
    learning_rate=0.000025,
    verbose=1,
    batch_size=32,
    train_freq=4,
    target_update_interval=500,
    learning_starts=500,
    buffer_size=25000,
    device="cuda",
    tensorboard_log="./tb_logs/",
)

saved_model_path = sys.argv[1]
model.load(saved_model_path)

evaluate_policy(model, env)