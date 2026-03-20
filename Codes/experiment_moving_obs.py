import pybullet as p
import pybullet_data
import time
import numpy as np

# --- 1. SETUP ---
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)
p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=45, cameraPitch=-30, cameraTargetPosition=[0.3, 0, 0.2])

# --- 2. MODELS & COLORS ---
p.loadURDF("plane.urdf")
robot = p.loadURDF("franka_panda/panda.urdf", [0, 0, 0], useFixedBase=True)

blocks = [
    p.loadURDF("cube_small.urdf", [0.4, 0.15, 0.05]), 
    p.loadURDF("cube_small.urdf", [0.4, 0.35, 0.05])  
]
bin_places = [[0.3, -0.25, 0.05], [0.3, -0.40, 0.05]]

# The "Drone" Obstacle
obstacle = p.loadURDF("sphere_small.urdf", [0.25, 0.1, 0.3], globalScaling=1.5)

p.changeVisualShape(blocks[0], -1, rgbaColor=[0.2, 0.6, 1.0, 1])  
p.changeVisualShape(blocks[1], -1, rgbaColor=[0.2, 1.0, 0.2, 1])  
p.changeVisualShape(obstacle, -1, rgbaColor=[1.0, 0, 0, 1]) 

ready_pos = [0, -0.5, 0, -2.0, 0, 1.5, 0.7]
for i in range(7):
    p.resetJointState(robot, i, ready_pos[i])

# --- 3. THE REACTIVE BRAIN ---
def get_reactive_vel(ee_pos, goal_pos, obs_pos, state):
    f_att = (np.array(goal_pos) - np.array(ee_pos)) * 4.5 
    
    dist_vec = np.array(ee_pos) - np.array(obs_pos)
    dist = np.linalg.norm(dist_vec)
    f_rep = np.array([0.0, 0.0, 0.0])
    
    # MODIFICATION: Slightly increased repulsion (0.012) and detection radius (0.25)
    if dist < 0.25 and state not in ["DESCEND", "GRASP", "DESCEND_PLACE"]: 
        # Repulsive Force
        f_push = (dist_vec / dist**3) * 0.012
        
        # Sidestep (Helps 'curve' the trajectory)
        f_side = np.array([-dist_vec[1], dist_vec[0], 0.08]) * 0.02
        f_rep = f_push + f_side
        
        # Red line shows the "Repulsion" force vector
        p.addUserDebugLine(ee_pos, ee_pos + (f_rep * 0.4), [1, 0, 0], 3, lifeTime=0.06) 

    # Green line shows the "Attraction" force vector
    p.addUserDebugLine(ee_pos, ee_pos + (f_att * 0.1), [0, 1, 0], 2, lifeTime=0.06) 

    total_vel = f_att + f_rep
    mag = np.linalg.norm(total_vel)
    limit = 0.07
    if mag > limit:
        total_vel = (total_vel / mag) * limit
    return total_vel

# --- 4. CONTROL VARIABLES ---
state = "APPROACH"
ee_index = 11
grasp_constraint = None
home_orn = p.getLinkState(robot, ee_index)[1] 
home_target = p.getLinkState(robot, ee_index)[0] 
prev_ee_pos = p.getLinkState(robot, ee_index)[0]

obs_target_pos = [0.4, 0, 0.3]
last_obs_update = time.time()
hover_duration = 3.0
current_idx = 0 

# --- 5. MAIN LOOP ---
while True:
    # Drone Logic
    now = time.time()
    if now - last_obs_update > hover_duration:
        obs_target_pos = [np.random.uniform(0.3, 0.5), np.random.uniform(-0.2, 0.3), np.random.uniform(0.15, 0.4)]
        last_obs_update = now
        hover_duration = np.random.uniform(2.0, 4.0)

    curr_obs_pos, _ = p.getBasePositionAndOrientation(obstacle)
    drone_vel = (np.array(obs_target_pos) - np.array(curr_obs_pos)) * 0.05
    p.resetBasePositionAndOrientation(obstacle, np.array(curr_obs_pos) + drone_vel, [0, 0, 0, 1])

    ee_pos = p.getLinkState(robot, ee_index)[0]
    obs_pos, _ = p.getBasePositionAndOrientation(obstacle)

    if current_idx < len(blocks):
        block_pos, _ = p.getBasePositionAndOrientation(blocks[current_idx])
        bin_hover = [bin_places[current_idx][0], bin_places[current_idx][1], 0.25]

    # --- STATE MACHINE ---
    if state == "APPROACH":
        target = [block_pos[0], block_pos[1], 0.35] 
        if np.linalg.norm(np.array(ee_pos) - np.array(target)) < 0.03: state = "HOVER_PICK"

    elif state == "HOVER_PICK":
        target = [block_pos[0], block_pos[1], 0.12] 
        if np.linalg.norm(np.array(ee_pos) - np.array(target)) < 0.02: state = "DESCEND"

    elif state == "DESCEND":
        target = [block_pos[0], block_pos[1], 0.01] 
        if np.linalg.norm(np.array(ee_pos) - np.array(target)) < 0.01: state = "GRASP"

    elif state == "GRASP":
        for f in [9, 10]: p.setJointMotorControl2(robot, f, p.POSITION_CONTROL, 0.01, force=30) 
        for _ in range(20): p.stepSimulation()
        grasp_constraint = p.createConstraint(robot, ee_index, blocks[current_idx], -1, p.JOINT_FIXED, [0, 0, 0], [0, 0, 0], [0, 0, 0])
        state = "LIFT"

    elif state == "LIFT":
        target = [ee_pos[0], ee_pos[1], 0.35] 
        if ee_pos[2] > 0.32: state = "GO_TO_BIN"

    elif state == "GO_TO_BIN":
        target = bin_hover
        if np.linalg.norm(np.array(ee_pos) - np.array(target)) < 0.05: state = "DESCEND_PLACE"

    elif state == "DESCEND_PLACE":
        target = bin_places[current_idx]
        if np.linalg.norm(np.array(ee_pos) - np.array(target)) < 0.02: state = "RELEASE"

    elif state == "RELEASE":
        if grasp_constraint: p.removeConstraint(grasp_constraint)
        for f in [9, 10]: p.setJointMotorControl2(robot, f, p.POSITION_CONTROL, 0.04)
        for _ in range(20): p.stepSimulation()
        state = "RETREAT"

    elif state == "RETREAT":
        target = [ee_pos[0], ee_pos[1], 0.35]
        if ee_pos[2] > 0.3:
            current_idx += 1
            state = "APPROACH" if current_idx < len(blocks) else "RETURN_HOME"

    elif state == "RETURN_HOME":
        target = home_target
        if np.linalg.norm(np.array(ee_pos) - np.array(target)) < 0.05:
            state = "DONE"
            print("Mission Accomplished!")

    # --- APPLY MOTION & TRACING ---
    if state != "DONE":
        vel = get_reactive_vel(ee_pos, target, obs_pos, state)
        next_pos = np.array(ee_pos) + vel
        joint_poses = p.calculateInverseKinematics(robot, ee_index, next_pos, targetOrientation=home_orn)
        for i in range(7):
            p.setJointMotorControl2(robot, i, p.POSITION_CONTROL, joint_poses[i], force=500)
        
        # DRAW THE TRAJECTORY (Persistent Path)
        p.addUserDebugLine(prev_ee_pos, ee_pos, [0, 0.8, 0], 3, lifeTime=15)
        prev_ee_pos = ee_pos

    p.stepSimulation()
    time.sleep(1./240.)