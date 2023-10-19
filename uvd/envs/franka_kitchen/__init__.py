import os
import sys

os.environ[
    "LD_LIBRARY_PATH"
] = f":{os.environ['HOME']}/.mujoco/mujoco210/bin:/usr/lib/nvidia"

# workaround to import adept envs
ADEPT_DIR = os.path.join(
    os.path.dirname(__file__), "relay-policy-learning", "adept_envs"
)
assert os.path.exists(ADEPT_DIR), ADEPT_DIR
sys.path.append(ADEPT_DIR)

from .franka_kitchen_base import *
from .franka_kitchen_constants import *

import adept_envs.mujoco_env

adept_envs.mujoco_env.USE_DM_CONTROL = USE_DM_CONTROL
