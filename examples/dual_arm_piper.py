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

def _create_base(root):
    # Create a tabletop base with a visual table, collision plane, and two attachment sites.
    base = root.worldbody.add("body", name="base")
    
    base.add(
        "geom",
        name="table_box",
        type="box",
        size=[0.6, 0.77, 0.04],
        density=1e-3,
        rgba=".9 .8 .6 1",
        pos=[0.4, 0, -0.04]
    )
    
    base.add(
        "geom",
        name="plane_above",
        type="plane",
        size=[0.6, 0.77, 0.01],
        pos=[0.4, 0, -0.001],
        rgba="0.7 0.7 0.7 0.9",
        contype=1,
        conaffinity=1
    )
    
    left_site = base.add(
        "site",
        name="piper_left_base",
        pos=[0.0, 0.35, 0],
        quat="1 0 0 0",
        group=5
    )
    right_site = base.add(
        "site",
        name="piper_right_base",
        pos=[0, -0.35, 0],
        quat="1 0 0 0",
        group=5
    )
    return left_site, right_site

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

def _add_arm(root, site, side):
    # Load and attach a Piper arm to the given site.
    arm = mjcf.from_path(str(_PIPER_XML))
    arm.model = f"piper_{side}"
    key = arm.find("key", "home")
    if key:
        key.remove()
    site.attach(arm)
    return arm

def _add_target(root, name, color):
    # Create a mocap target with a box geometry.
    target = root.worldbody.add("body", name=name, mocap=True)
    target.add(
        "geom",
        type="box",
        size=[0.05, 0.05, 0.05],
        contype=0,
        conaffinity=0,
        rgba=color
    )
    return target

def _create_limits(model):
    # Set collision and velocity limits.
    limits = [mink.ConfigurationLimit(model=model)]
    collision_pairs = [
        (["piper_left/piper_end_effector"], ["plane_above"]),
        (["piper_right/piper_end_effector"], ["plane_above"])
    ]
    limits.append(mink.CollisionAvoidanceLimit(model=model, geom_pairs=collision_pairs))

    joint_velocities = {
        "joint1": 3.1416,
        "joint2": 3.4034,
        "joint3": 3.1416,
        "joint4": 3.92699,
        "joint5": 3.92699,
        "joint6": 3.92699,
    }
    max_velocities = {
        f"piper_{side}/{joint}": vel
        for side in ["left", "right"]
        for joint, vel in joint_velocities.items()
    }
    limits.append(mink.VelocityLimit(model, max_velocities))
    return limits

def construct_model():
    # Build the complete model with dual Piper arms on a tabletop.
    root = mjcf.RootElement()
    root.statistic.meansize = 0.08
    getattr(root.visual, "global").azimuth = -120
    getattr(root.visual, "global").elevation = -20
    root.worldbody.add("light", pos="0 0 1.5", directional="true")

    add_world_origin_marker(root)
    
    left_site, right_site = _create_base(root)
    
    _add_arm(root, left_site, "left")
    _add_arm(root, right_site, "right")
    
    _add_target(root, "l_target", ".3 .6 .3 .5")
    _add_target(root, "r_target", ".3 .3 .6 .5")
    
    # Optionally, add a world frame site.
    root.worldbody.add("site", name="world_frame", pos=[0, 0, 0], quat="1 0 0 0", size="0.05", group=7)
    
    return mujoco.MjModel.from_xml_string(root.to_xml_string(), root.get_assets())

def _create_tasks(model):
    # Define IK tasks for each arm and a posture task.
    tasks = [
        mink.FrameTask(
            frame_name=f"piper_{side}/piper_end_effector",
            frame_type="body",
            position_cost=1.0,
            orientation_cost=0.2,
            lm_damping=1.0,
        ) for side in ["left", "right"]
    ]
    tasks.append(mink.PostureTask(model=model, cost=0.0))
    return tasks

def _set_home_config(data):
    # Set the initial joint positions for both arms.
    qpos = [
        0, 1.57, -1.3485, 0, 0, 0,   # left arm (6 joints)
        0, 1.57, -1.3485, 0, 0, 0    # right arm (6 joints)
    ]
    if len(data.qpos) != len(qpos):
        raise ValueError(f"Expected {len(qpos)} qpos, got {len(data.qpos)}")
    data.qpos[:] = qpos

if __name__ == "__main__":
    model = construct_model()
    configuration = mink.Configuration(model)
    data = configuration.data

    tasks = _create_tasks(model)
    _set_home_config(data)
    limits = _create_limits(model)
    configuration.update(data.qpos)
    tasks[-1].set_target_from_configuration(configuration)  # Posture task

    with mujoco.viewer.launch_passive(model=model, data=data, show_left_ui=False, show_right_ui=False) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)
        viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE

        # Initialize mocap targets.
        for mocap, frame in zip(
            ["l_target", "r_target"],
            ["piper_left/piper_end_effector", "piper_right/piper_end_effector"]
        ):
            mink.move_mocap_to_frame(model, data, mocap, frame, "body")

        rate = RateLimiter(frequency=500.0, warn=False)
        while viewer.is_running():
            # Update IK targets.
            for task, target in zip(tasks[:2], ["l_target", "r_target"]):
                task.set_target(mink.SE3.from_mocap_name(model, data, target))
            
            # Solve IK and integrate.
            vel = mink.solve_ik(configuration, tasks, rate.dt, "quadprog", 1e-3, limits=limits)
            configuration.integrate_inplace(vel, rate.dt)
            
            # Render.
            mujoco.mj_camlight(model, data)
            viewer.sync()
            rate.sleep()
