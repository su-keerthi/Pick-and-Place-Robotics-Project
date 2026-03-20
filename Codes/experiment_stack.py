import pybullet as p
import pybullet_data
import time
import numpy as np
import random

# ------------------ SETUP ------------------
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)

# High solver iterations for stacking stability
p.setPhysicsEngineParameter(
    numSolverIterations=500,
    contactBreakingThreshold=0.0001,
    fixedTimeStep=1/240.0
)

p.resetDebugVisualizerCamera(1.8, 0, -45, [0, 0, 0])
p.loadURDF("plane.urdf")
robot = p.loadURDF("franka_panda/panda.urdf", [0, 0, 0], useFixedBase=True)
ee_index = 11

# Panda joint limits
ll = [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973]
ul = [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973]
jr = [5.8, 3.5, 5.8, 3.1, 5.8, 3.8, 5.8]

# ------------------ BLOCKS & TARGETS ------------------
def random_position(existing):
    while True:
        radius = random.uniform(0.5, 0.65)
        angle = random.uniform(-np.pi/4, np.pi/4) # Keep in front of robot
        pos = [radius * np.cos(angle), radius * np.sin(angle), 0.025]
        if all(np.linalg.norm(np.array(pos[:2]) - np.array(e[:2])) > 0.15 for e in existing):
            return pos

block_positions = []
for _ in range(3):
    block_positions.append(random_position(block_positions))

blocks = [p.loadURDF("cube_small.urdf", pos) for pos in block_positions]
colors = [[0.2, 0.6, 1, 1], [0.2, 1, 0.2, 1], [1, 1, 0, 1]]
for i, b in enumerate(blocks):
    p.changeVisualShape(b, -1, rgbaColor=colors[i])

stack_center = [0.5, -0.2]
# Blocks are 0.05m tall. Z centers: 0.025, 0.075, 0.125
stack_targets = [[stack_center[0], stack_center[1], 0.025 + (i * 0.05)] for i in range(3)]

# ------------------ HELPERS ------------------
def reactive_velocity(ee_pos, goal, obstacles, last_vel):
    f_att = (np.array(goal) - np.array(ee_pos)) * 5.0
    f_rep = np.array([0.0, 0.0, 0.0])
    
    for obs in obstacles:
        obs_pos, _ = p.getBasePositionAndOrientation(obs)
        diff = np.array(ee_pos) - np.array(obs_pos)
        d = np.linalg.norm(diff)
        if d < 0.25:
            f_rep += (diff / d**3) * 0.05
            
    raw = f_att + f_rep
    smooth = (0.2 * raw) + (0.8 * last_vel)
    mag = np.linalg.norm(smooth)
    limit = 0.08
    if mag > limit: smooth = (smooth / mag) * limit
    return smooth

# ------------------ INITIAL STATE ------------------
ready = [0, -0.5, 0, -2.0, 0, 1.5, 0.7]
for i in range(7): p.resetJointState(robot, i, ready[i])

state = "APPROACH"
current_block = 0
grasp_constraint = None
current_vel = np.array([0.0, 0.0, 0.0])
home_orn = p.getQuaternionFromEuler([0, np.pi, 0]) # Top-down grip

# ------------------ MAIN LOOP ------------------
while current_block < 3:
    ee_pos = p.getLinkState(robot, ee_index)[0]
    block = blocks[current_block]
    b_pos, b_orn = p.getBasePositionAndOrientation(block)
    dest = stack_targets[current_block]

    if state == "APPROACH":
        target = [b_pos[0], b_pos[1], 0.25]
        for j in [9, 10]: p.setJointMotorControl2(robot, j, p.POSITION_CONTROL, 0.04)
        if np.linalg.norm(np.array(ee_pos) - np.array(target)) < 0.02:
            state = "DESCEND"

    elif state == "DESCEND":
        target = [b_pos[0], b_pos[1], b_pos[2] + 0.01] # Hover slightly above center
        if np.linalg.norm(np.array(ee_pos) - np.array(target)) < 0.01:
            state = "GRASP"

    elif state == "GRASP":
        # Close fingers
        for j in [9, 10]: p.setJointMotorControl2(robot, j, p.POSITION_CONTROL, 0.0, force=200)
        for _ in range(50): p.stepSimulation()
        
        # Lock block to gripper
        grasp_constraint = p.createConstraint(robot, ee_index, block, -1, p.JOINT_FIXED, 
                                            [0, 0, 0], [0, 0, 0], [0, 0, 0.01])
        state = "LIFT"

    elif state == "LIFT":
        target = [ee_pos[0], ee_pos[1], 0.3]
        if ee_pos[2] > 0.25: state = "MOVE_STACK"

    elif state == "MOVE_STACK":
        target = [dest[0], dest[1], 0.3]
        if np.linalg.norm(np.array(ee_pos[:2]) - np.array(target[:2])) < 0.01:
            state = "PLACE"

    elif state == "PLACE":
        # Target is the exact Z-height calculated
        target = [dest[0], dest[1], dest[2] + 0.015] 
        if np.linalg.norm(np.array(ee_pos) - np.array(target)) < 0.005:
            current_vel *= 0 # Stop movement
            state = "RELEASE"

    elif state == "RELEASE":
        if grasp_constraint:
            p.removeConstraint(grasp_constraint)
            grasp_constraint = None
        
        for j in [9, 10]: p.setJointMotorControl2(robot, j, p.POSITION_CONTROL, 0.04)
        for _ in range(100): p.stepSimulation() # Wait for physics to settle
        state = "RETREAT"

    elif state == "RETREAT":
        target = [ee_pos[0], ee_pos[1], 0.4]
        if ee_pos[2] > 0.35:
            current_block += 1
            state = "APPROACH"

    # Execution logic
    current_vel = reactive_velocity(ee_pos, target, [], current_vel)
    next_pos = np.array(ee_pos) + current_vel
    
    joint_poses = p.calculateInverseKinematics(robot, ee_index, next_pos, home_orn,
                                               ll, ul, jr, ready, maxNumIterations=200)
    
    for i in range(7):
        p.setJointMotorControl2(robot, i, p.POSITION_CONTROL, joint_poses[i], force=500)

    p.stepSimulation()
    time.sleep(1/240)

print("Stacking Complete.")