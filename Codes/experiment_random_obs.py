import pybullet as p
import pybullet_data  
import time
import numpy as np
import random

# --- 1. SETUP ---
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)
p.resetDebugVisualizerCamera(cameraDistance=2.0, cameraYaw=45, cameraPitch=-40, cameraTargetPosition=[0, 0, 0])

p.loadURDF("plane.urdf")
robot = p.loadURDF("franka_panda/panda.urdf", [0, 0, 0], useFixedBase=True)

ll = [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973]
ul = [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973]
jr = [5.8, 3.5, 5.8, 3.1, 5.8, 3.8, 5.8]
rp = [0, -0.5, 0, -2.0, 0, 1.5, 0.7]
for i in range(7): p.resetJointState(robot, i, rp[i])

# --- 2. POSES ---
pick_pos = [0.4, -0.4, 0.025]
place_pos = [-0.4, -0.4, 0.025]
home_pos = [0.3, 0.0, 0.4]

block = p.loadURDF("cube_small.urdf", pick_pos)
p.changeVisualShape(block, -1, rgbaColor=[0.2, 0.6, 1.0, 1])

# --- 3. WIDE BOARD SCATTER ---
obstacles = []
shapes = [p.GEOM_BOX, p.GEOM_SPHERE, p.GEOM_CYLINDER]
num_obstacles = 40 

for _ in range(num_obstacles):
    shape_type = random.choice(shapes)
    color = [random.random(), random.random(), random.random(), 0.9]
    
    if shape_type == p.GEOM_BOX:
        extents = [random.uniform(0.02, 0.05) for _ in range(3)] 
        col_id = p.createCollisionShape(shape_type, halfExtents=extents)
        vis_id = p.createVisualShape(shape_type, halfExtents=extents, rgbaColor=color)
    elif shape_type == p.GEOM_SPHERE:
        radius = random.uniform(0.03, 0.06) 
        col_id = p.createCollisionShape(shape_type, radius=radius)
        vis_id = p.createVisualShape(shape_type, radius=radius, rgbaColor=color)
    else: 
        radius = random.uniform(0.02, 0.05) 
        cyl_h = random.uniform(0.08, 0.2) 
        col_id = p.createCollisionShape(shape_type, radius=radius, height=cyl_h)
        vis_id = p.createVisualShape(shape_type, radius=radius, length=cyl_h, rgbaColor=color)

    while True:
        obs_x = random.uniform(-1.2, 1.2) 
        obs_y = random.uniform(-1.2, 1.2)
        obs_z = random.uniform(0.05, 0.35) 
        
        dist_to_pick = np.linalg.norm(np.array([obs_x, obs_y]) - np.array(pick_pos[:2]))
        dist_to_place = np.linalg.norm(np.array([obs_x, obs_y]) - np.array(place_pos[:2]))
        dist_to_home = np.linalg.norm(np.array([obs_x, obs_y]) - np.array(home_pos[:2]))
        dist_to_base = np.linalg.norm(np.array([obs_x, obs_y]))
        
        if dist_to_pick > 0.20 and dist_to_place > 0.20 and dist_to_home > 0.20 and dist_to_base > 0.20:
            break 

    obs = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col_id, baseVisualShapeIndex=vis_id, basePosition=[obs_x, obs_y, obs_z])
    obstacles.append(obs)

# --- 4. APF WITH ABSOLUTE ESCAPE OVERRIDE ---
def get_reactive_velocity(robot_id, ee_index, target_pos, obstacle_ids, escape_vec):
    ee_state = p.getLinkState(robot_id, ee_index)
    ee_pos = np.array(ee_state[0])

    if np.linalg.norm(escape_vec) > 0:
        return escape_vec

    f_att = (np.array(target_pos) - np.array(ee_pos)) * 2.0
    if np.linalg.norm(f_att) > 1.0: f_att = (f_att / np.linalg.norm(f_att)) * 1.0
    
    f_rep = np.array([0.0, 0.0, 0.0])
    influence_radius = 0.05
    
    for obs in obstacle_ids:
        closest_points = p.getClosestPoints(bodyA=robot_id, bodyB=obs, distance=influence_radius)
        if closest_points:
            valid_points = [pt for pt in closest_points if pt[3] > 0] 
            if valid_points:
                pt = min(valid_points, key=lambda x: x[8])
                dist = max(pt[8], 0.005) 
                
                if dist < influence_radius:
                    normal = np.array(pt[7]) 
                    tangent = np.cross(normal, np.array([0, 0, 1])) 
                    
                    magnitude = 0.02 * (1.0/dist - 1.0/influence_radius) / (dist**2)
                    f_rep += (normal + tangent * 0.4) * magnitude
                    p.addUserDebugLine(pt[6], pt[6] + (normal * 0.1), [1, 0, 0], 3.0, lifeTime=0.1)

    desired_vel = f_att + f_rep
    speed = np.linalg.norm(desired_vel)
    if speed > 0.08: desired_vel = (desired_vel / speed) * 0.08 
        
    return desired_vel

# --- 5. EXECUTION VARIABLES ---
state = "APPROACH"
ee_index = 11
grasp_constraint = None
home_orn = p.getLinkState(robot, ee_index)[1] 

state_start_time = time.time()
escape_vector = np.array([0.0, 0.0, 0.0])
escape_timeout = 0
# Initial Debug Text ID
debug_text_id = p.addUserDebugText(f"STATE: {state}", [0, 0, 1.4], textColorRGB=[1, 1, 1])

def change_state(new_state):
    global state, state_start_time
    state = new_state
    state_start_time = time.time()
    print(f"State -> {state}")

# --- 6. MAIN LOOP ---
while True:
    ee_pos = p.getLinkState(robot, ee_index)[0]
    current_time = time.time()

    # Update Debug Text in Real-time
    p.addUserDebugText(f"STATE: {state}", [0, 0, 1.4], textColorRGB=[1, 1, 1], replaceItemUniqueId=debug_text_id)

    # --- TRUE ESCAPE TRIGGER ---
    if state in ["APPROACH", "GO_TO_PLACE", "RETURN_HOME"]:
        if current_time - state_start_time > 4.5:
            print(">>> Path Blocked! Forcing Absolute Escape Override...")
            escape_timeout = current_time + 1.2 
            state_start_time = current_time + 1.2 
            
            dx = random.uniform(-1.0, 1.0)
            dy = random.uniform(-1.0, 1.0)
            dz = random.uniform(1.0, 2.5) 
            vec = np.array([dx, dy, dz])
            escape_vector = (vec / np.linalg.norm(vec)) * 0.15 

    if current_time > escape_timeout:
        escape_vector = np.array([0.0, 0.0, 0.0])

    # --- STATE MACHINE ---
    if state == "APPROACH":
        target = [pick_pos[0], pick_pos[1], 0.35]
        if np.linalg.norm(np.array(ee_pos) - np.array(target)) < 0.04: change_state("DESCEND")

    elif state == "DESCEND":
        target = [pick_pos[0], pick_pos[1], 0.025] 
        if np.linalg.norm(np.array(ee_pos) - np.array(target)) < 0.015: change_state("GRASP")

    elif state == "GRASP":
        for f in [9, 10]: p.setJointMotorControl2(robot, f, p.POSITION_CONTROL, 0.0, force=100)
        for _ in range(60): p.stepSimulation(); time.sleep(1./240.)
        grasp_constraint = p.createConstraint(robot, ee_index, block, -1, p.JOINT_FIXED, [0, 0, 0], [0, 0, 0], [0, 0, 0])
        change_state("LIFT")

    elif state == "LIFT":
        target = [pick_pos[0], pick_pos[1], 0.45] 
        if ee_pos[2] > 0.4: change_state("GO_TO_PLACE")

    elif state == "GO_TO_PLACE":
        target = [place_pos[0], place_pos[1], 0.45]
        if np.linalg.norm(np.array(ee_pos) - np.array(target)) < 0.04: change_state("DESCEND_PLACE")

    elif state == "DESCEND_PLACE":
        target = [place_pos[0], place_pos[1], place_pos[2] + 0.02] 
        if np.linalg.norm(np.array(ee_pos) - np.array(target)) < 0.015: change_state("RELEASE")

    elif state == "RELEASE":
        if grasp_constraint: p.removeConstraint(grasp_constraint); grasp_constraint = None
        for f in [9, 10]: p.setJointMotorControl2(robot, f, p.POSITION_CONTROL, 0.04)
        for _ in range(60): p.stepSimulation(); time.sleep(1./240.)
        change_state("RETREAT_UP")

    elif state == "RETREAT_UP":
        target = [place_pos[0], place_pos[1], 0.45]
        if ee_pos[2] > 0.4: change_state("RETURN_HOME")

    elif state == "RETURN_HOME":
        target = home_pos
        if np.linalg.norm(np.array(ee_pos) - np.array(target)) < 0.04: 
            # MISSION COMPLETE: Update UI and kill the loop
            p.addUserDebugText("STATUS: DONE (LOCKED)", [0, 0, 1.4], textColorRGB=[0, 1, 0], replaceItemUniqueId=debug_text_id)
            print("Mission Complete! Robot home and locked.")
            change_state("DONE")
            break # <--- THIS STOPS THE ROBOT AND KEEPS THE WINDOW OPEN

    # --- REACTIVE MOTION CONTROLLER ---
    if state != "DONE":
        active_obs = obstacles if state in ["APPROACH", "GO_TO_PLACE", "RETURN_HOME"] else []
        
        vel = get_reactive_velocity(robot, ee_index, target, active_obs, escape_vector)
        next_pos = np.array(ee_pos) + vel
        
        if next_pos[2] < 0.02: next_pos[2] = 0.02
        
        joint_poses = p.calculateInverseKinematics(
            robot, ee_index, next_pos, targetOrientation=home_orn, 
            lowerLimits=ll, upperLimits=ul, jointRanges=jr, restPoses=rp, maxNumIterations=100
        )
        for i in range(7):
            p.setJointMotorControl2(robot, i, p.POSITION_CONTROL, joint_poses[i], force=500)

    p.stepSimulation()
    time.sleep(1./240.)

# This keeps the script alive so you can inspect the window after the loop breaks
print("Robot has reached terminal state. Press Ctrl+C in terminal or close window to exit.")
while True:
    p.stepSimulation()
    time.sleep(1./240.)