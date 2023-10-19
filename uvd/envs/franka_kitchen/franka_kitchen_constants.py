import numpy as np

USE_DM_CONTROL = False
BONUS_THRESH = 0.3

FRANKA_KITCHEN_ALL_TASKS = [
    "bottom burner",
    "top burner",
    "light switch",
    "slide cabinet",
    "hinge cabinet",
    "microwave",
    "kettle",
]

OBS_ELEMENT_INDICES = {
    # rotation & opening for bottom left burner
    "bottom burner": np.array([11, 12]),
    # rotation & opening for top left burner
    "top burner": np.array([15, 16]),
    # joint angle & opening
    "light switch": np.array([17, 18]),
    # Translation of the slide cabinet joint
    "slide cabinet": np.array([19]),
    # Rotation of the joint in the (left, right) hinge cabinet
    "hinge cabinet": np.array([20, 21]),
    # Rotation of the joint in the microwave door
    "microwave": np.array([22]),
    # x, y, z, qx, qy, qz, qw
    "kettle": np.array([23, 24, 25, 26, 27, 28, 29]),
}

OBS_ELEMENT_GOALS = {
    "bottom burner": np.array([-0.88, -0.01]),
    "top burner": np.array([-0.92, -0.01]),
    "light switch": np.array([-0.69, -0.05]),
    "slide cabinet": np.array([0.37]),
    # right hinge, left should be (-1.45, 0)
    "hinge cabinet": np.array([0.0, 1.45]),
    "microwave": np.array([-0.75]),
    "kettle": np.array([-0.23, 0.75, 1.62, 0.99, 0.0, 0.0, -0.06]),
}

OBS_ELEMENT_THRESH = {
    "bottom burner": 0.31,
    "top burner": 0.31,
    "light switch": 0.3,  # 0.1
    "slide cabinet": 0.2,
    "hinge cabinet": 0.2,
    "microwave": 0.2,
    # "kettle": np.array([0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.2]),
    "kettle": np.array([0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.3]),
}

ELEMENT_TO_IDX = {k: i for i, k in enumerate(OBS_ELEMENT_GOALS.keys())}
IDX_TO_ELEMENT = {i: k for k, i in ELEMENT_TO_IDX.items()}

FRANKA_KITCHEN_REWARD_CONFIG = {
    "progress": 0.0,  # complete sub-goal
    "terminal": 10.0,  # complete all goals
    "intrinsic_weight": 1.0,  # intrinsic reward if using embedding-dist-diff reward
}
