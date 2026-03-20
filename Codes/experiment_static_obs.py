
import pybullet as p
import pybullet_data
import time
import numpy as np

# --- 1. SETUP ---
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)
p.resetDebugVisualizerCamera(cameraDistance=1.2, cameraYaw=45, cameraPitch=-30, cameraTargetPosition=[0.3, 0, 0])

# --- 2. MODELS & COLORS ---
p.loadURDF("plane.urdf")
robot = p.loadURDF("franka_panda/panda.urdf", [0, 0, 0], useFixedBase=True)

blocks = [
    p.loadURDF("cube_small.urdf", [0.4, 0.15, 0.05]), 
    p.loadURDF("cube_small.urdf", [0.4, 0.35, 0.05])  
]
bin_places = [
    [0.3, -0.25, 0.02], 
    [0.3, -0.40, 0.02] 
]

obstacle = p.loadURDF("sphere_small.urdf", [0.25, 0.1, 0.15], globalScaling=2.5)

p.changeVisualShape(blocks[0], -1, rgbaColor=[0.2, 0.6, 1.0, 1])  
p.changeVisualShape(blocks[1], -1, rgbaColor=[0.2, 1.0, 0.2, 1])  
p.changeVisualShape(obstacle, -1, rgbaColor=[1.0, 0.4, 0.0, 0.9]) 

ready_pos = [0, -0.5, 0, -2.0, 0, 1.5, 0.7]
for i in range(7):
    p.resetJointState(robot, i, ready_pos[i])

# --- 3. THE REACTIVE BRAIN ---
def get_reactive_vel(ee_pos, goal_pos, obs_pos):
    dist_to_goal = np.linalg.norm(np.array(goal_pos) - np.array(ee_pos))
    f_att = (np.array(goal_pos) - np.array(ee_pos)) * 3.5 
    dist_to_obs = np.linalg.norm(np.array(ee_pos) - np.array(obs_pos))
    f_rep = np.array([0.0, 0.0, 0.0])
    
    if dist_to_obs < 0.25: 
        f_push = (np.array(ee_pos) - np.array(obs_pos)) / dist_to_obs**3 * 0.015
        f_side = np.array([-(ee_pos[1]-obs_pos[1]), (ee_pos[0]-obs_pos[0]), 0]) * 0.02
        f_rep = (f_push + f_side) * dist_to_goal 
    
    p.addUserDebugLine(ee_pos, ee_pos + (f_att * 0.1), [0, 1, 0], 2.5, lifeTime=0.1) 
    if np.linalg.norm(f_rep) > 0:
        p.addUserDebugLine(ee_pos, ee_pos + (f_rep * 0.1), [1, 0, 0], 2.5, lifeTime=0.1) 

    total_vel = f_att + f_rep
    mag = np.linalg.norm(total_vel)
    if mag > 0.06: 
        total_vel = (total_vel / mag) * 0.06
    return total_vel

# --- 4. CONTROL VARIABLES ---
state = "APPROACH"
ee_index = 11
grasp_constraint = None
lift_x, lift_y = 0.0, 0.0
home_orn = p.getLinkState(robot, ee_index)[1] 
home_target = p.getLinkState(robot, ee_index)[0] 

current_idx = 0 
num_blocks = len(blocks)

# --- 5. MAIN LOOP ---
while True:
    new_obs_y = 0.1 + np.sin(time.time() * 2.0) * 0.1 
    p.resetBasePositionAndOrientation(obstacle, [0.25, new_obs_y, 0.15], [0, 0, 0, 1])

    ee_pos = p.getLinkState(robot, ee_index)[0]
    obs_pos, _ = p.getBasePositionAndOrientation(obstacle)

    if current_idx < num_blocks:
        current_block = blocks[current_idx]
        block_pos, _ = p.getBasePositionAndOrientation(current_block)
        current_bin = bin_places[current_idx]
        bin_hover = [current_bin[0], current_bin[1], 0.2]

    # --- STATE MACHINE ---
    if state == "APPROACH":
        target = [block_pos[0], block_pos[1], 0.35] 
        if np.linalg.norm(np.array(ee_pos) - np.array(target)) < 0.03:
            lift_x, lift_y = ee_pos[0], ee_pos[1]
            state = "HOVER_STABILIZE"

    elif state == "HOVER_STABILIZE":
        target = [lift_x, lift_y, 0.35]
        p.stepSimulation()
        time.sleep(0.5) 
        state = "HOVER_PICK"

    elif state == "HOVER_PICK":
        target = [lift_x, lift_y, 0.12] 
        if np.linalg.norm(np.array(ee_pos) - np.array(target)) < 0.02:
            state = "DESCEND"

    elif state == "DESCEND":
        target = [lift_x, lift_y, 0.005] 
        if np.linalg.norm(np.array(ee_pos) - np.array(target)) < 0.01:
            state = "GRASP"

    elif state == "GRASP":
        for f in [9, 10]:
            p.setJointMotorControl2(robot, f, p.POSITION_CONTROL, 0.01, force=30) 
        p.stepSimulation()
        time.sleep(0.5) 
        grasp_constraint = p.createConstraint(robot, ee_index, current_block, -1, p.JOINT_FIXED, [0, 0, 0], [0, 0, 0], [0, 0, 0])
        state = "LIFT"

    elif state == "LIFT":
        target = [lift_x, lift_y, 0.35] 
        if ee_pos[2] > 0.32:
            state = "GO_TO_BIN"

    elif state == "GO_TO_BIN":
        target = bin_hover
        if np.linalg.norm(np.array(ee_pos) - np.array(target)) < 0.05:
            lift_x, lift_y = ee_pos[0], ee_pos[1]
            state = "DESCEND_PLACE"

    elif state == "DESCEND_PLACE":
        target = [lift_x, lift_y, current_bin[2]]
        if np.linalg.norm(np.array(ee_pos) - np.array(target)) < 0.02:
            state = "RELEASE"

    elif state == "RELEASE":
        if grasp_constraint is not None:
            p.removeConstraint(grasp_constraint)
            grasp_constraint = None
        for f in [9, 10]:
            p.setJointMotorControl2(robot, f, p.POSITION_CONTROL, 0.04)
        p.stepSimulation()
        time.sleep(0.5) 
        state = "RETREAT_UP"

    elif state == "RETREAT_UP":
        target = [lift_x, lift_y, 0.35] 
        if ee_pos[2] > 0.3:
            current_idx += 1
            state = "APPROACH" if current_idx < num_blocks else "RETURN_HOME"

    elif state == "RETURN_HOME":
        target = home_target
        if np.linalg.norm(np.array(ee_pos) - np.array(target)) < 0.05:
            print("Mission Accomplished!")
            state = "DONE"

    # --- APPLY MOTION ---
    if state != "DONE":
        if state in ["HOVER_STABILIZE", "HOVER_PICK", "DESCEND", "GRASP", "LIFT", "DESCEND_PLACE", "RELEASE", "RETREAT_UP"]:
            fake_obs = [100.0, 100.0, 100.0] 
            vel = get_reactive_vel(ee_pos, target, fake_obs)
        else:
            vel = get_reactive_vel(ee_pos, target, obs_pos)
            
        next_pos = np.array(ee_pos) + vel
        
        # KEY FIX: Changed maxIter to maxNumIterations
        joint_poses = p.calculateInverseKinematics(robot, ee_index, next_pos, targetOrientation=home_orn, maxNumIterations=100)
        
        for i in range(7):
            p.setJointMotorControl2(robot, i, p.POSITION_CONTROL, joint_poses[i], force=500)

    p.stepSimulation()
    time.sleep(1./240.)