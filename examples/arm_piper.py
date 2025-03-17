from pathlib import Path
import mujoco
import mujoco.viewer
import numpy as np
import mink
from dm_control import mjcf
from loop_rate_limiters import RateLimiter

# Define file paths.
_HERE = Path(__file__).parent
_PIPER_XML = _HERE / "agilex_piper" / "piper.xml"

# Build the scene.
root = mjcf.RootElement()
root.statistic.meansize = 0.08
getattr(root.visual, "global").azimuth = 120
getattr(root.visual, "global").elevation = -20
root.worldbody.add("light", pos=[0, 0, 1.5], directional=True)

# Add a floor.
root.worldbody.add(
    "geom",
    name="floor",
    type="plane",
    size=[0.5, 0.5, 0.05],
    pos=[0, 0, 0],
    rgba=".8 .8 .8 0.2",
    contype=1,
    conaffinity=1
)

# Create a base with an attachment site.
base = root.worldbody.add("body", name="base", pos=[0, 0, 0])
attachment_site = base.add(
    "site",
    name="piper_attachment_site",
    pos=[0, 0, 0],
    quat="1 0 0 0",
    size="0.01",
    group=5
)

# Load and attach the Piper arm.
arm = mjcf.from_path(str(_PIPER_XML))
key = arm.find("key", "home")
if key:
    key.remove()
attachment_site.attach(arm)

# Add a mocap target for controlling the end-effector.
target = root.worldbody.add("body", name="target", mocap=True, pos=[0.5, 0, 0.5])
target.add(
    "geom",
    type="box",
    size=[0.05, 0.05, 0.05],
    contype=0,
    conaffinity=0,
    rgba="0.6 0.3 0.3 0.2"
)

# Build the Mujoco model.
model = mujoco.MjModel.from_xml_string(root.to_xml_string(), root.get_assets())

# Create configuration and obtain simulation data.
configuration = mink.Configuration(model)
data = configuration.data

# Define the IK task for the end-effector.
tasks = [
    mink.FrameTask(
        frame_name="piper_description/piper_end_effector",
        frame_type="body",
        position_cost=1.0,
        orientation_cost=1.0,
        lm_damping=1.0,
    )
]

# Set the home configuration.
qpos_home = [0, 1.57, -1.3485, 0, 0, 0]
if len(data.qpos) != len(qpos_home):
    raise ValueError(f"Expected {len(qpos_home)} qpos, got {len(data.qpos)}")
data.qpos[:] = qpos_home
configuration.update(data.qpos)

# Set up velocity limits.
joint_velocities = {
    "joint1": 3.1416,
    "joint2": 3.4034,
    "joint3": 3.1416,
    "joint4": 3.92699,
    "joint5": 3.92699,
    "joint6": 3.92699,
}
max_velocities = {f"piper_description/{joint}": vel * 1.5 for joint, vel in joint_velocities.items()}
velocity_limit = mink.VelocityLimit(model, max_velocities)

# Launch the Mujoco viewer.
with mujoco.viewer.launch_passive(model=model, data=data, show_left_ui=False, show_right_ui=False) as viewer:
    mujoco.mjv_defaultFreeCamera(model, viewer.cam)
    viewer.opt.frame = mujoco.mjtFrame.mjFRAME_NONE
    # Initialize the mocap target from the current end-effector frame.
    mink.move_mocap_to_frame(model, data, "target", "piper_description/piper_end_effector", "body")
    
    rate = RateLimiter(frequency=500.0, warn=False)
    while viewer.is_running():
        T_wt = mink.SE3.from_mocap_name(model, data, "target")
        tasks[0].set_target(T_wt)
        vel = mink.solve_ik(configuration, tasks, rate.dt, "quadprog", 1e-3, limits=[velocity_limit])
        configuration.integrate_inplace(vel, rate.dt)
        mujoco.mj_camlight(model, data)
        viewer.sync()
        rate.sleep()
