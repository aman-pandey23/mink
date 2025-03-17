from pathlib import Path
import mujoco
import mujoco.viewer
from dm_control import mjcf
from loop_rate_limiters import RateLimiter
import mink

# Define file paths.
_HERE = Path(__file__).parent
_XML = _HERE / "franka_fr3" / "fr3.xml"

def _create_base_single(root):
    # Create a static base with a square table and a collision plane.
    base = root.worldbody.add("body", name="base")
    base.add(
        "geom",
        name="table_box",
        type="box",
        size=[0.5, 0.5, 0.04],
        density=1e-3,
        rgba=".9 .8 .6 1",
        pos=[0, 0, -0.04]
    )
    base.add(
        "geom",
        name="plane_above",
        type="plane",
        size=[0.5, 0.5, 0.01],
        pos=[0, 0, 0.0],
        rgba="0.7 0.7 0.7 0.15",
        contype=1,
        conaffinity=1
    )
    # Create a central attachment site.
    site = base.add(
        "site",
        name="fr3_left_base",
        pos=[0, 0, 0],
        quat="1 0 0 0",
        group=5
    )
    return site

def add_world_origin_marker(root):
    # Add a marker at the world origin.
    root.worldbody.add(
        "site",
        name="world_origin",
        pos="0 0 0",
        size="0.03",
        type="sphere",
        rgba="1 0 0 1"
    )

def _add_arm(root, site, side="left"):
    # Load and attach a FR3 arm to the specified site.
    arm = mjcf.from_path(_XML.as_posix())
    arm.model = f"fr3_{side}"
    key = arm.find("key", "home")
    if key:
        key.remove()
    site.attach(arm)
    return arm

def _add_target(root, name="target", color=".3 .6 .3 .5"):
    # Create a mocap target body.
    target = root.worldbody.add("body", name=name, mocap=True)
    target.add(
        "geom",
        type="box",
        size=".05 .05 .05",
        contype="0",
        conaffinity="0",
        rgba=color
    )
    return target

def construct_model():
    # Build the MJCF model with a single FR3 arm.
    root = mjcf.RootElement()
    root.statistic.meansize = 0.08
    getattr(root.visual, "global").azimuth = -120
    getattr(root.visual, "global").elevation = -20
    root.worldbody.add("light", pos="0 0 1.5", directional="true")
    add_world_origin_marker(root)
    site = _create_base_single(root)
    _add_arm(root, site, side="left")
    _add_target(root, name="target", color=".3 .6 .3 .5")
    return mujoco.MjModel.from_xml_string(root.to_xml_string(), root.get_assets())

def _create_task(model):
    # Define the IK task for the arm's end-effector.
    task = mink.FrameTask(
        frame_name="fr3_left/attachment_site",
        frame_type="site",
        position_cost=1.0,
        orientation_cost=1.0,
        lm_damping=1.0,
    )
    posture_task = mink.PostureTask(model=model, cost=0.05)
    return [task, posture_task]

def _create_limits(model):
    # Set velocity and collision avoidance constraints.
    limits = [mink.ConfigurationLimit(model=model)]
    collision_pairs = [
        (["fr3_left/fr3_link7_collision"], ["plane_above"])
    ]
    limits.append(mink.CollisionAvoidanceLimit(model=model, geom_pairs=collision_pairs))
    joint_velocities = {
        "fr3_joint1": 2.62,
        "fr3_joint2": 2.62,
        "fr3_joint3": 2.62,
        "fr3_joint4": 2.62,
        "fr3_joint5": 5.26,
        "fr3_joint6": 5.26,
        "fr3_joint7": 5.26,
    }
    max_velocities = {f"fr3_left/{joint}": vel for joint, vel in joint_velocities.items()}
    limits.append(mink.VelocityLimit(model, max_velocities))
    return limits

def _set_home_config(data):
    # Initialize joint positions for the FR3 arm.
    qpos = [1.57079, 0, 0, -1.57079, 0, 1.57079, -0.7853]
    if len(data.qpos) != len(qpos):
        raise ValueError(f"Expected {len(qpos)} qpos, got {len(data.qpos)}")
    data.qpos[:] = qpos

if __name__ == "__main__":
    model = construct_model()
    configuration = mink.Configuration(model)
    data = configuration.data
    tasks = _create_task(model)
    limits = _create_limits(model)
    _set_home_config(data)
    configuration.update(data.qpos)
    tasks[-1].set_target_from_configuration(configuration)
    
    with mujoco.viewer.launch_passive(
        model=model, data=data, show_left_ui=False, show_right_ui=False
    ) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)
        viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE
        mink.move_mocap_to_frame(model, data, "target", "fr3_left/attachment_site", "site")
        rate = RateLimiter(frequency=500.0, warn=False)
        while viewer.is_running():
            tasks[0].set_target(mink.SE3.from_mocap_name(model, data, "target"))
            vel = mink.solve_ik(configuration, tasks, rate.dt, "quadprog", 1e-3, limits=limits)
            configuration.integrate_inplace(vel, rate.dt)
            mujoco.mj_camlight(model, data)
            viewer.sync()
            rate.sleep()
