import pybullet as p
import pybullet_data
import time
import numpy as np
import random 

# --- 1. SETUP ---
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)
p.setPhysicsEngineParameter(numSolverIterations=100) 
p.resetDebugVisualizerCamera(cameraDistance=2.2, cameraYaw=45, cameraPitch=-40, cameraTargetPosition=[0, 0, 0])

# --- 2. MODELS & ENVIRONMENT ---
p.loadURDF("plane.urdf")
robot = p.loadURDF("franka_panda/panda.urdf", [0, 0, 0], useFixedBase=True)

# Panda IK parameters
ll = [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973]
ul = [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973]
jr = [5.8, 3.5, 5.8, 3.1, 5.8, 3.8, 5.8]

# Seed angles for the IK solver
unravel_angles = [0, -0.5, 0, -2.0, 0, 1.5, 0.7]   # Neutral / Approach
elbow_up_angles = [0, -1.2, 0, -1.2, 0, 1.2, 0.7]  # Arched / Carrying

def get_random_pos(existing):
    while True:
        r, a = random.uniform(0.5, 0.8), random.uniform(0, 2 * np.pi)
        pos = [r * np.cos(a), r * np.sin(a), 0.05]
        if all(np.linalg.norm(np.array(pos[:2]) - np.array(p[:2])) > 0.15 for p in existing): 
            return pos

# --- SPAWN 12 BLOCKS ---
blocks, block_colors, block_positions = [], [], []
color_map = {"BLUE": [0.2, 0.6, 1.0, 1], "GREEN": [0.2, 1.0, 0.2, 1], "YELLOW": [1.0, 1.0, 0.0, 1]}
colors_to_spawn = ["BLUE", "GREEN", "YELLOW"] * 4 

for color_name in colors_to_spawn:
    pos = get_random_pos(block_positions)
    bid = p.loadURDF("cube_small.urdf", pos)
    p.changeVisualShape(bid, -1, rgbaColor=color_map[color_name])
    blocks.append(bid); block_colors.append(color_name); block_positions.append(pos)

drop_zone_centers = {"BLUE": [0.5, 0.4, 0.02], "GREEN": [0.5, -0.4, 0.02], "YELLOW": [-0.6, 0.0, 0.02]}
zone_counters = {"BLUE": 0, "GREEN": 0, "YELLOW": 0}

for i in range(7): p.resetJointState(robot, i, unravel_angles[i])

# --- 3. MOTION HELPERS ---
state, ee_index, current_idx = "APPROACH", 11, 0
home_orn = p.getLinkState(robot, ee_index)[1]
home_target = [0.2, 0.0, 0.6] 
grasp_constraint = None 
state_start_time = time.time()
debug_text_id = p.addUserDebugText("STARTING...", [0, 0, 1.5], textColorRGB=[1, 1, 1])

def change_state(new_state):
    global state, state_start_time
    state = new_state
    state_start_time = time.time()
    print(f"State -> {state}")

def move_robot(target_xyz, use_elbow_up=False):
    """IK Solver seeded with a specific bias to prevent self-collision."""
    bias_pose = elbow_up_angles if use_elbow_up else unravel_angles
    joint_poses = p.calculateInverseKinematics(
        robot, ee_index, target_xyz, targetOrientation=home_orn, 
        lowerLimits=ll, upperLimits=ul, jointRanges=jr, restPoses=bias_pose
    )
    for i in range(7):
        p.setJointMotorControl2(robot, i, p.POSITION_CONTROL, joint_poses[i], force=500)

# --- 4. MAIN LOOP ---
while True:
    ee_pos = p.getLinkState(robot, ee_index)[0]
    current_time = time.time()

    status = f"STATE: {state} | BLOCK: {current_idx+1}/12"
    p.addUserDebugText(status, [0, 0, 1.5], textColorRGB=[1,1,1], replaceItemUniqueId=debug_text_id)

    if current_idx < len(blocks):
        c_block, c_color = blocks[current_idx], block_colors[current_idx]
        block_pos, _ = p.getBasePositionAndOrientation(c_block)
        base, count = drop_zone_centers[c_color], zone_counters[c_color]
        dest_pos = [base[0] + (count // 2) * 0.08, base[1] + (count % 2) * 0.08, base[2]]
    else: 
        state = "RETURN_HOME"

    # --- WATCHDOG ---
    if state in ["APPROACH", "DESCEND", "GO_TO_ZONE", "DESCEND_PLACE"]:
        if current_time - state_start_time > 4.5:
            change_state("UNRAVEL")

    # --- STATE MACHINE ---
    if state == "UNRAVEL":
        for i in range(7): p.setJointMotorControl2(robot, i, p.POSITION_CONTROL, unravel_angles[i], force=500)
        curr_j = [p.getJointState(robot, i)[0] for i in range(7)]
        if np.linalg.norm(np.array(curr_j) - np.array(unravel_angles)) < 0.15: change_state("APPROACH")

    elif state == "APPROACH":
        # HARD CLEAR
        if grasp_constraint is not None:
            p.removeConstraint(grasp_constraint); grasp_constraint = None
        for j in [9, 10]: p.setJointMotorControl2(robot, j, p.POSITION_CONTROL, 0.04)
        
        target = [block_pos[0], block_pos[1], 0.35]
        move_robot(target)
        if np.linalg.norm(np.array(ee_pos) - np.array(target)) < 0.03: change_state("DESCEND")

    elif state == "DESCEND":
        target = [block_pos[0], block_pos[1], 0.03]
        move_robot(target)
        if np.linalg.norm(np.array(ee_pos) - np.array(target)) < 0.015: change_state("GRASP")

    elif state == "GRASP":
        for f in [9, 10]: p.setJointMotorControl2(robot, f, p.POSITION_CONTROL, 0.0, force=200)
        for _ in range(40): p.stepSimulation(); time.sleep(1./240.)
        grasp_constraint = p.createConstraint(robot, ee_index, c_block, -1, p.JOINT_FIXED, [0,0,0], [0,0,0], [0,0,0])
        change_state("LIFT")

    elif state == "LIFT":
        target = [ee_pos[0], ee_pos[1], 0.5] 
        move_robot(target, use_elbow_up=True) #
        if ee_pos[2] > 0.45: change_state("GO_TO_ZONE")

    elif state == "GO_TO_ZONE":
        target = [dest_pos[0], dest_pos[1], 0.5]
        move_robot(target, use_elbow_up=True)
        if np.linalg.norm(np.array(ee_pos) - np.array(target)) < 0.05: change_state("DESCEND_PLACE")

    elif state == "DESCEND_PLACE":
        target = [dest_pos[0], dest_pos[1], dest_pos[2] + 0.06]
        move_robot(target, use_elbow_up=True)
        if np.linalg.norm(np.array(ee_pos) - np.array(target)) < 0.03: change_state("RELEASE")

    elif state == "RELEASE":
        if grasp_constraint is not None:
            p.removeConstraint(grasp_constraint); grasp_constraint = None #
        for f in [9, 10]: p.setJointMotorControl2(robot, f, p.POSITION_CONTROL, 0.04)
        for _ in range(60): p.stepSimulation(); time.sleep(1./240.)
        zone_counters[c_color] += 1
        change_state("RETREAT")

    elif state == "RETREAT":
        target = [ee_pos[0], ee_pos[1], 0.5]
        move_robot(target)
        if ee_pos[2] > 0.45:
            current_idx += 1; change_state("APPROACH")

    elif state == "RETURN_HOME":
        move_robot(home_target)
        if np.linalg.norm(np.array(ee_pos) - np.array(home_target)) < 0.05: break

    p.stepSimulation()
    time.sleep(1./240.)