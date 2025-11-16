import pybullet as p
import pybullet_data
import time
import numpy as np
from pick_place import create_graspable_cube, attach_object_to_robot, detach_object, move_to_position

# Debug flags
DEBUG_COLLISION_MONITORING = True  # Monitor and report collisions

# 1️⃣ Connect to PyBullet
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)

# 2️⃣ Load environment
plane = p.loadURDF("plane.urdf")

# Load table (white cube)
table_height = 0.4
table_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.5, 0.5, table_height/2])
table_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.5, 0.5, table_height/2], 
                                    rgbaColor=[1, 1, 1, 1])
table = p.createMultiBody(baseMass=0, 
                        baseCollisionShapeIndex=table_collision,
                        baseVisualShapeIndex=table_visual,
                        basePosition=[0.5, 0, table_height/2])

# 3️⃣ Load UR5 robot on top of table (with 2cm buffer)
ur5 = p.loadURDF("../urdf/ur5.urdf", 
                 basePosition=[0.5, 0, table_height + 0.02],
                 baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
                 useFixedBase=True)

print(f"UR5 loaded with {p.getNumJoints(ur5)} joints")

# Set initial joint positions (stable upright pose)
initial_positions = [0, -90, 90, 180, -90, 0]
initial_positions = [np.deg2rad(pos) for pos in initial_positions]
for i in range(6):
    p.resetJointState(ur5, i, initial_positions[i])

# Enable strong position control to hold against gravity
for i in range(6):
    p.setJointMotorControl2(
        bodyIndex=ur5,
        jointIndex=i,
        controlMode=p.POSITION_CONTROL,
        targetPosition=initial_positions[i],
        force=5000,
        maxVelocity=2.0,
        positionGain=0.5
    )

# 4️⃣ Create a small cube to pick up
cube_size = 0.04
table_top_z = table_height + 0.02
cube_position = [0.8, -0.3, table_top_z + cube_size/2 + 0.001]
cube = create_graspable_cube(
    position=cube_position,
    size=cube_size,
    color=[1, 0, 0, 1],
    mass=0.05
)

print(f"\nCube created at position: {cube_position}")
print(f"Robot base at (0.5, 0, {table_height + 0.02})")

# Let physics settle
for _ in range(100):
    p.stepSimulation()

# Get initial end-effector position and orientation
num_joints = p.getNumJoints(ur5)
ee_link = num_joints - 1
initial_ee_state = p.getLinkState(ur5, ee_link)
initial_ee_pos = initial_ee_state[0]
initial_ee_orn = initial_ee_state[1]
print(f"Initial end-effector position: {initial_ee_pos}")
print(f"Initial end-effector orientation: {initial_ee_orn}")

# Get cube position after settling
cube_pos, _ = p.getBasePositionAndOrientation(cube)
print(f"Cube settled at: {cube_pos}")


# Collision monitoring function
def check_and_report_collisions(robot_id, obstacle_ids, frame_count):
    """Monitor and report collisions without interfering with motion."""
    if not DEBUG_COLLISION_MONITORING:
        return
    
    # Check self-collision
    p.performCollisionDetection()
    self_contacts = p.getContactPoints(bodyA=robot_id, bodyB=robot_id)
    
    if len(self_contacts) > 0:
        print(f"\n⚠️  SELF-COLLISION DETECTED at frame {frame_count}!")
        for contact in self_contacts[:3]:  # Show first 3
            link_a = contact[3]
            link_b = contact[4]
            print(f"   Link {link_a} colliding with Link {link_b}")
    
    # Check collision with obstacles
    for obstacle_id in obstacle_ids:
        contacts = p.getContactPoints(bodyA=robot_id, bodyB=obstacle_id)
        if len(contacts) > 0:
            obstacle_name = "Table" if obstacle_id == obstacle_ids[0] else "Plane"
            print(f"\n⚠️  COLLISION WITH {obstacle_name.upper()} at frame {frame_count}!")
            for contact in contacts[:3]:
                link_id = contact[3]
                contact_force = contact[9]  # Normal force
                print(f"   Link {link_id}, Force: {contact_force:.2f}N")


# 5️⃣ Pick and return to initial position demo
print("\n=== Pick and Return to Initial Position Demo ===")
print("Collision monitoring ENABLED\n")

hover_height = 0.05
pick_pos = [cube_pos[0], cube_pos[1], cube_pos[2]]
hover_pick_pos = [pick_pos[0], pick_pos[1], pick_pos[2] + hover_height]

frame_counter = 0

# Step 1: Move to hover above cube (pointing down)
print("1. Moving to hover position above cube...")
if not move_to_position(ur5, hover_pick_pos, obstacle_ids=[table, plane], speed_factor=1.0):
    print("✗ Failed to reach hover position!")
else:
    time.sleep(0.5)
    
    # Step 2: Move down to pick (pointing down)
    print("2. Moving down to pick up cube...")
    if not move_to_position(ur5, pick_pos, obstacle_ids=[table, plane], speed_factor=0.5):
        print("✗ Failed to reach pick position!")
    else:
        time.sleep(0.5)
        
        # Step 3: Grasp the cube
        print("3. Grasping cube...")
        constraint_id = attach_object_to_robot(ur5, cube, ee_link)
        time.sleep(0.5)
        
        # Step 4: Lift up (pointing down)
        print("4. Lifting cube...")
        if not move_to_position(ur5, hover_pick_pos, obstacle_ids=[table, plane], speed_factor=0.5):
            print("✗ Failed to lift!")
        else:
            time.sleep(0.5)
            
            # Step 5: Return to initial position with cube (use initial orientation)
            print("5. Returning to initial position with cube...")
            if not move_to_position(ur5, initial_ee_pos, target_orn=initial_ee_orn, obstacle_ids=[table, plane], speed_factor=1.0):
                print("✗ Failed to return to initial position!")
            else:
                print("\n✓ Successfully picked up cube and returned to initial position!")
                time.sleep(1.0)
                
                # Keep simulation running with collision monitoring
                print("\n=== Collision Monitoring Active ===")
                print("Monitoring for self-collisions and table/plane collisions...")
                print("Close window to exit.\n")
                
                while True:
                    p.stepSimulation()
                    frame_counter += 1
                    
                    # Check collisions every 60 frames (1 second at 60Hz)
                    if frame_counter % 60 == 0:
                        check_and_report_collisions(ur5, [table, plane], frame_counter)
                    
                    time.sleep(1./60.)
