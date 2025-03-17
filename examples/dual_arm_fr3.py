from pathlib import Path
import mujoco
import mujoco.viewer
from dm_control import mjcf
from loop_rate_limiters import RateLimiter
import mink

# Define file paths.
_HERE = Path(__file__).parent
_XML = _HERE / "franka_fr3" / "fr3.xml"

def _create_base(root):
    # Create a static base with table, collision plane, and attachment sites.
    base = root.worldbody.add("body", name="base")
    
    base.add(
        "geom",
        name="table_box",
        type="box",
        size=[0.6, 0.77, 0.04],
        density=1e-3,
        rgba=".9 .8 .6 1",
        pos=[0, 0, -0.035]
    )
    base.add(
        "geom",
        name="plane_above",
        type="plane",
        size=[0.6, 0.77, 0.01],
        pos=[0.1, 0, 0.004],
        rgba="0.7 0.7 0.7 0.15",
        contype=1,
        conaffinity=1
    )
    
    # Create left and right attachment sites.
    sites = {
        "left": (0, 0.5, 0, "0.7071 0 0 -0.7071"),
        "right": (0, -0.5, 0, "0.7071 0 0 0.7071")
    }
    
    for side, (x, y, z, quat) in sites.items():
        site = base.add(
            "site",
            name=f"fr3_{side}_base",
            pos=[x, y, z],
            quat=quat,
            group=5
        )
        yield site

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
    # Load and attach the FR3 arm to the given site.
    arm = mjcf.from_path(_XML.as_posix())
    arm.model = f"fr3_{side}"
    arm.find("key", "home").remove()
    site.attach(arm)
    return arm

def _add_target(root, name, color):
    # Create a mocap target with visual geometry.
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
    # Build the complete MJCF model with dual FR3 arms.
    root = mjcf.RootElement()
    root.statistic.meansize = 0.08
    getattr(root.visual, "global").azimuth = -120
    getattr(root.visual, "global").elevation = -20
    root.worldbody.add("light", pos="0 0 1.5", directional="true")

    add_world_origin_marker(root)

    # Create base and attachment sites.
    left_site, right_site = _create_base(root)
    
    # Attach arms.
    _add_arm(root, left_site, "left")
    _add_arm(root, right_site, "right")

    # Add mocap targets.
    _add_target(root, "l_target", ".3 .6 .3 .5")
    _add_target(root, "r_target", ".3 .3 .6 .5")

    return mujoco.MjModel.from_xml_string(root.to_xml_string(), root.get_assets())

def _create_tasks(model):
    # Initialize IK tasks for both end-effectors and posture.
    tasks = [
        mink.FrameTask(
            frame_name=f"fr3_{side}/attachment_site",
            frame_type="site",
            position_cost=1.0,
            orientation_cost=1.0,
            lm_damping=1.0,
        ) for side in ["left", "right"]
    ]
    tasks.append(mink.PostureTask(model=model, cost=0.1))
    return tasks

def _create_limits(model):
    # Set velocity and collision avoidance constraints.
    limits = [mink.ConfigurationLimit(model=model)]
    
    collision_pairs = [
        (["fr3_left/fr3_link7_collision", "fr3_right/fr3_link7_collision"], ["plane_above"])
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
    max_velocities = {
        f"fr3_{side}/{joint}": vel
        for side in ["left", "right"]
        for joint, vel in joint_velocities.items()
    }
    limits.append(mink.VelocityLimit(model, max_velocities))
    
    return limits

def _set_home_config(data):
    # Set initial joint positions for both arms.
    qpos = [
        1.57079, 0, 0, -1.57079, 0, 1.57079, -0.7853,  # Left arm
        -1.57079, 0, 0, -1.57079, 0, 1.57079, -0.7853   # Right arm
    ]
    if len(data.qpos) != 14:
        raise ValueError(f"Expected 14 qpos, got {len(data.qpos)}")
    data.qpos[:] = qpos

if __name__ == "__main__":
    model = construct_model()
    configuration = mink.Configuration(model)
    data = configuration.data

    tasks = _create_tasks(model)
    limits = _create_limits(model)
    _set_home_config(data)
    configuration.update(data.qpos)
    tasks[-1].set_target_from_configuration(configuration)  # Posture task

    with mujoco.viewer.launch_passive(
        model=model, data=data, show_left_ui=False, show_right_ui=False
    ) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)
        viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE

        # Initialize mocap targets.
        for mocap, frame in zip(
            ["l_target", "r_target"],
            ["fr3_left/attachment_site", "fr3_right/attachment_site"]
        ):
            mink.move_mocap_to_frame(model, data, mocap, frame, "site")

        rate = RateLimiter(frequency=500.0, warn=False)
        while viewer.is_running():
            # Update IK targets for each arm.
            for task, target in zip(tasks[:2], ["l_target", "r_target"]):
                task.set_target(mink.SE3.from_mocap_name(model, data, target))
            
            # Solve IK and integrate.
            vel = mink.solve_ik(configuration, tasks, rate.dt, "quadprog", 1e-3, limits=limits)
            configuration.integrate_inplace(vel, rate.dt)
            
            # Render and update.
            mujoco.mj_camlight(model, data)
            viewer.sync()
            rate.sleep()
