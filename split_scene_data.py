# import os
# import cv2
# import random
# import string
# import numpy as np
# import habitat_sim

# from tqdm import tqdm
# from pyquaternion import Quaternion
# from habitat_sim.utils.common import quat_from_two_vectors, quat_from_angle_axis

# # Function to generate a random string or use a scene-specific name
# def generate_folder_name(scene_name, idx):
#     random_suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=16))
#     return f"{random_suffix}"

# # Function to normalize quaternion
# def normalize_quaternion(quat):
#     norm = np.linalg.norm([quat.x, quat.y, quat.z, quat.w])
#     return Quaternion(quat.x / norm, quat.y / norm, quat.z / norm, quat.w / norm)

# # Function to create the simulator configuration
# def make_cfg(scene_glb, sensor_height, width, height):
#     sim_cfg = habitat_sim.SimulatorConfiguration()
#     sim_cfg.gpu_device_id = 0
#     sim_cfg.scene_id = scene_glb
#     sim_cfg.enable_physics = False

#     # Sensor configuration
#     sensor_specs = []

#     # RGB Sensor
#     color_sensor_spec = habitat_sim.CameraSensorSpec()
#     color_sensor_spec.uuid = "color_sensor"
#     color_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
#     color_sensor_spec.resolution = [height, width]
#     color_sensor_spec.position = [0.0, sensor_height, 0.0]
#     color_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
#     sensor_specs.append(color_sensor_spec)

#     # Agent configuration
#     agent_cfg = habitat_sim.agent.AgentConfiguration()
#     agent_cfg.sensor_specifications = sensor_specs
#     agent_cfg.action_space = {
#         "move_forward": habitat_sim.agent.ActionSpec(
#             "move_forward", habitat_sim.agent.ActuationSpec(amount=0.25)
#         ),
#         "turn_left": habitat_sim.agent.ActionSpec(
#             "turn_left", habitat_sim.agent.ActuationSpec(amount=30.0)
#         ),
#         "turn_right": habitat_sim.agent.ActionSpec(
#             "turn_right", habitat_sim.agent.ActuationSpec(amount=30.0)
#         ),
#     }

#     return habitat_sim.Configuration(sim_cfg, [agent_cfg])

# # Function to sample points along the path
# def sample_points_along_path(path_points, interval=1.0):
#     sampled_positions = []
#     accumulated_distance = 0.0
#     last_point = path_points[0]
#     sampled_positions.append(last_point)

#     for point in path_points[1:]:
#         segment = point - last_point
#         segment_length = np.linalg.norm(segment)
#         while accumulated_distance + segment_length >= interval:
#             remaining_distance = interval - accumulated_distance
#             ratio = remaining_distance / segment_length
#             new_point = last_point + ratio * segment
#             sampled_positions.append(new_point)
#             last_point = new_point
#             segment = point - last_point
#             segment_length = np.linalg.norm(segment)
#             accumulated_distance = 0.0
#         accumulated_distance += segment_length
#         last_point = point

#     return sampled_positions

# # Function to sample two different paths
# def sample_two_different_paths(sim, num_samples=200):
#     navigable_points = []

#     # Sample multiple navigable points
#     for _ in range(num_samples):
#         point = sim.pathfinder.get_random_navigable_point()
#         navigable_points.append(point)

#     # Define the paths
#     path1 = habitat_sim.ShortestPath()
#     path2 = habitat_sim.ShortestPath()

#     # Variables to store the farthest and second farthest points
#     max_distance = 0
#     best_path_2_score = -float('inf')
#     start_point_1 = None
#     end_point_1 = None
#     start_point_2 = None
#     end_point_2 = None

#     # Nested loop to find the farthest points for path 1
#     for i in range(len(navigable_points)):
#         for j in range(i + 1, len(navigable_points)):
#             dist = np.linalg.norm(navigable_points[i] - navigable_points[j])

#             # Check if path between points is navigable (geodesic distance is finite)
#             temp_path = habitat_sim.ShortestPath()
#             temp_path.requested_start = navigable_points[i]
#             temp_path.requested_end = navigable_points[j]
#             if not sim.pathfinder.find_path(temp_path) or temp_path.geodesic_distance == float('inf'):
#                 continue  # Skip if points are not navigable

#             # Update the farthest points for the first path
#             if dist > max_distance:
#                 max_distance = dist
#                 start_point_1 = navigable_points[i]
#                 end_point_1 = navigable_points[j]

#     # Find path 2: it should be as different from path 1 as possible
#     for i in range(len(navigable_points)):
#         for j in range(i + 1, len(navigable_points)):
#             # Use np.array_equal to compare arrays correctly
#             if np.array_equal(navigable_points[i], start_point_1) or np.array_equal(navigable_points[i], end_point_1) or \
#                np.array_equal(navigable_points[j], start_point_1) or np.array_equal(navigable_points[j], end_point_1):
#                 # Skip points that overlap with path 1
#                 continue

#             # Check if path between points is navigable (geodesic distance is finite)
#             temp_path = habitat_sim.ShortestPath()
#             temp_path.requested_start = navigable_points[i]
#             temp_path.requested_end = navigable_points[j]
#             if not sim.pathfinder.find_path(temp_path) or temp_path.geodesic_distance == float('inf'):
#                 continue  # Skip if points are not navigable

#             # Distance between points in path 2
#             dist_2 = np.linalg.norm(navigable_points[i] - navigable_points[j])

#             # Distance from path 2 points to path 1 points (measure dissimilarity)
#             dist_start_to_path1 = min(np.linalg.norm(navigable_points[i] - start_point_1), np.linalg.norm(navigable_points[i] - end_point_1))
#             dist_end_to_path1 = min(np.linalg.norm(navigable_points[j] - start_point_1), np.linalg.norm(navigable_points[j] - end_point_1))

#             # The score favors distant points from path 1 and long distances in path 2
#             path_2_score = dist_2 + dist_start_to_path1 + dist_end_to_path1

#             # Update the second path if this one is more different from path 1
#             if path_2_score > best_path_2_score:
#                 best_path_2_score = path_2_score
#                 start_point_2 = navigable_points[i]
#                 end_point_2 = navigable_points[j]

#     # Set the points for the two paths
#     path1.requested_start = start_point_1
#     path1.requested_end = end_point_1
#     path2.requested_start = start_point_2
#     path2.requested_end = end_point_2

#     # Ensure paths are navigable
#     if not sim.pathfinder.find_path(path1) or path1.geodesic_distance == float('inf'):
#         print(f"Path 1 is not navigable.")
#         return None  # Handle the case where the first path is not valid

#     if not sim.pathfinder.find_path(path2) or path2.geodesic_distance == float('inf'):
#         print(f"Path 2 is not navigable.")
#         return None  # Handle the case where the second path is not valid

#     return (start_point_1, end_point_1), (start_point_2, end_point_2), path1, path2

# # Function to save images and poses into a new folder for each sampled point
# def save_data_in_folder(output_dir, scene_name, agent_positions, path_number, sim, agent):
#     for idx, position in tqdm(enumerate(agent_positions), desc=f"Path {path_number}"):
#         # Create a unique folder for each sampled point
#         folder_name = generate_folder_name(scene_name, idx)
#         point_folder = os.path.join(output_dir, folder_name)
#         os.makedirs(point_folder, exist_ok=True)

#         # Set agent state to the current position
#         agent_state = habitat_sim.AgentState()
#         agent_state.position = position
#         agent.set_state(agent_state)

#         # Randomly sample 12 angles between 30° and 60°
#         rotations = np.random.uniform(30, 60, 12)

#         # Apply random rotations to the agent
#         for yaw_index, yaw in enumerate(rotations):

#             agent.agent_config.action_space["turn_right"].actuation.amount = yaw
#             agent.act("turn_right")

#             # Get observations after the action
#             observations = sim.get_sensor_observations()
#             rgb = observations["color_sensor"]

#             # Save the RGB image
#             rgb_filename = os.path.join(point_folder, f"{yaw_index:06d}.png")
#             cv2.imwrite(rgb_filename, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

#             # Save the pose data (position and quaternion)
#             pose_data = np.concatenate((agent.get_state().position, agent.get_state().rotation.components))
#             pose_filename = os.path.join(point_folder, f"{yaw_index:06d}.npy")
#             np.save(pose_filename, pose_data)

# # Main loop to process multiple scenes
# main_directories = [
#     "/home/shared/haoze/data/data/scene_datasets/hm3d/train",  # First directory
#     "/home/shared/haoze/data/scene_datasets/hm3d/minival",  # First directory
# ]

# # List of json files to exclude
# json_files = [
#     "hm3d_annotated_minival_basis.scene_dataset_config.json",
#     "hm3d_minival_basis.scene_dataset_config.json",
# ]

# for main_directory in main_directories:
#     for scene_dir in os.listdir(main_directory):
#         scene_path = os.path.join(main_directory, scene_dir)

#         # Skip json files and non-directories
#         if scene_dir in json_files or not os.path.isdir(scene_path):
#             continue

#         # Extract file prefix based on your directory structure
#         # Adjust this part if necessary
#         file_prefix = scene_dir.split("-")[-1]
#         scene_glb = os.path.join(scene_path, f"{file_prefix}.basis.glb")
#         navmesh_path = os.path.join(scene_path, f"{file_prefix}.basis.navmesh")

#         if not os.path.exists(scene_glb) or not os.path.exists(navmesh_path):
#             print(f"Skipping {scene_dir}: missing .glb or .navmesh file")
#             continue

#         output_dir = "./indoor_data"
#         os.makedirs(output_dir, exist_ok=True)

#         # Create simulator configuration
#         cfg = make_cfg(scene_glb, 1.5, 512, 512)

#         # Create simulator
#         sim = habitat_sim.Simulator(cfg)
#         sim.recompute_navmesh(sim.pathfinder, habitat_sim.NavMeshSettings())
#         sim.pathfinder.load_nav_mesh(navmesh_path)

#         # Initialize agent
#         agent = sim.initialize_agent(0)

#         # Sample two distinct paths
#         result = sample_two_different_paths(sim)
#         if result is None:
#             print(f"Could not find two navigable paths in {scene_dir}. Skipping.")
#             sim.close()
#             continue
#         (start_point_1, end_point_1), (start_point_2, end_point_2), path1, path2 = result

#         # Retrieve path points
#         path1_points = path1.points
#         path2_points = path2.points

#         # Get sampled points for both paths
#         agent_positions_1 = sample_points_along_path(path1_points, interval=0.5)
#         agent_positions_2 = sample_points_along_path(path2_points, interval=0.5)

#         # Save data for both paths
#         save_data_in_folder(output_dir, scene_dir, agent_positions_1, path_number=1, sim=sim, agent=agent)
#         save_data_in_folder(output_dir, scene_dir, agent_positions_2, path_number=2, sim=sim, agent=agent)

#         # Close simulator
#         sim.close()



import os
import cv2
import random
import string
import numpy as np
import habitat_sim

from tqdm import tqdm
from pyquaternion import Quaternion
from habitat_sim.utils.common import quat_from_two_vectors, quat_from_angle_axis

# Function to generate a random string or use a scene-specific name
def generate_folder_name(scene_name, idx):
    random_suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=16))
    return f"{random_suffix}"

# Function to normalize quaternion
def normalize_quaternion(quat):
    norm = np.linalg.norm([quat.x, quat.y, quat.z, quat.w])
    return Quaternion(quat.x / norm, quat.y / norm, quat.z / norm, quat.w / norm)

# Function to create the simulator configuration
def make_cfg(scene_glb, sensor_height, width, height):
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.gpu_device_id = 0
    sim_cfg.scene_id = scene_glb
    sim_cfg.enable_physics = False

    # Sensor configuration
    sensor_specs = []

    # RGB Sensor
    color_sensor_spec = habitat_sim.CameraSensorSpec()
    color_sensor_spec.uuid = "color_sensor"
    color_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
    color_sensor_spec.resolution = [height, width]
    color_sensor_spec.position = [0.0, sensor_height, 0.0]
    color_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    sensor_specs.append(color_sensor_spec)

    # Agent configuration
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = sensor_specs
    agent_cfg.action_space = {
        "move_forward": habitat_sim.agent.ActionSpec(
            "move_forward", habitat_sim.agent.ActuationSpec(amount=0.25)
        ),
        "turn_left": habitat_sim.agent.ActionSpec(
            "turn_left", habitat_sim.agent.ActuationSpec(amount=30.0)
        ),
        "turn_right": habitat_sim.agent.ActionSpec(
            "turn_right", habitat_sim.agent.ActuationSpec(amount=30.0)
        ),
    }

    return habitat_sim.Configuration(sim_cfg, [agent_cfg])

# Function to sample points along the path
def sample_points_along_path(path_points, interval=1.0):
    sampled_positions = []
    accumulated_distance = 0.0
    last_point = path_points[0]
    sampled_positions.append(last_point)

    for point in path_points[1:]:
        segment = point - last_point
        segment_length = np.linalg.norm(segment)
        while accumulated_distance + segment_length >= interval:
            remaining_distance = interval - accumulated_distance
            ratio = remaining_distance / segment_length
            new_point = last_point + ratio * segment
            sampled_positions.append(new_point)
            last_point = new_point
            segment = point - last_point
            segment_length = np.linalg.norm(segment)
            accumulated_distance = 0.0
        accumulated_distance += segment_length
        last_point = point

    return sampled_positions

# Function to sample two different paths
def sample_two_different_paths(sim, num_samples=200):
    navigable_points = []

    # Sample multiple navigable points
    for _ in range(num_samples):
        point = sim.pathfinder.get_random_navigable_point()
        navigable_points.append(point)

    # Define the paths
    path1 = habitat_sim.ShortestPath()
    path2 = habitat_sim.ShortestPath()

    # Variables to store the farthest and second farthest points
    max_distance = 0
    best_path_2_score = -float('inf')
    start_point_1 = None
    end_point_1 = None
    start_point_2 = None
    end_point_2 = None

    # Nested loop to find the farthest points for path 1
    for i in range(len(navigable_points)):
        for j in range(i + 1, len(navigable_points)):
            dist = np.linalg.norm(navigable_points[i] - navigable_points[j])

            # Check if path between points is navigable (geodesic distance is finite)
            temp_path = habitat_sim.ShortestPath()
            temp_path.requested_start = navigable_points[i]
            temp_path.requested_end = navigable_points[j]
            if not sim.pathfinder.find_path(temp_path) or temp_path.geodesic_distance == float('inf'):
                continue  # Skip if points are not navigable

            # Update the farthest points for the first path
            if dist > max_distance:
                max_distance = dist
                start_point_1 = navigable_points[i]
                end_point_1 = navigable_points[j]

    # Find path 2: it should be as different from path 1 as possible
    for i in range(len(navigable_points)):
        for j in range(i + 1, len(navigable_points)):
            # Use np.array_equal to compare arrays correctly
            if np.array_equal(navigable_points[i], start_point_1) or np.array_equal(navigable_points[i], end_point_1) or \
               np.array_equal(navigable_points[j], start_point_1) or np.array_equal(navigable_points[j], end_point_1):
                # Skip points that overlap with path 1
                continue

            # Check if path between points is navigable (geodesic distance is finite)
            temp_path = habitat_sim.ShortestPath()
            temp_path.requested_start = navigable_points[i]
            temp_path.requested_end = navigable_points[j]
            if not sim.pathfinder.find_path(temp_path) or temp_path.geodesic_distance == float('inf'):
                continue  # Skip if points are not navigable

            # Distance between points in path 2
            dist_2 = np.linalg.norm(navigable_points[i] - navigable_points[j])

            # Distance from path 2 points to path 1 points (measure dissimilarity)
            dist_start_to_path1 = min(np.linalg.norm(navigable_points[i] - start_point_1), np.linalg.norm(navigable_points[i] - end_point_1))
            dist_end_to_path1 = min(np.linalg.norm(navigable_points[j] - start_point_1), np.linalg.norm(navigable_points[j] - end_point_1))

            # The score favors distant points from path 1 and long distances in path 2
            path_2_score = dist_2 + dist_start_to_path1 + dist_end_to_path1

            # Update the second path if this one is more different from path 1
            if path_2_score > best_path_2_score:
                best_path_2_score = path_2_score
                start_point_2 = navigable_points[i]
                end_point_2 = navigable_points[j]

    # Set the points for the two paths
    path1.requested_start = start_point_1
    path1.requested_end = end_point_1
    path2.requested_start = start_point_2
    path2.requested_end = end_point_2

    # Ensure paths are navigable
    if not sim.pathfinder.find_path(path1) or path1.geodesic_distance == float('inf'):
        print(f"Path 1 is not navigable.")
        return None  # Handle the case where the first path is not valid

    if not sim.pathfinder.find_path(path2) or path2.geodesic_distance == float('inf'):
        print(f"Path 2 is not navigable.")
        return None  # Handle the case where the second path is not valid

    return (start_point_1, end_point_1), (start_point_2, end_point_2), path1, path2

# Function to save images and poses into a new folder for each sampled point
def save_data_in_folder(output_dir, scene_name, agent_positions, path_number, sim, agent, folder_counter, max_folders):
    angles = [0, 10, 20, 30]  # Fixed angles
    for idx, position in tqdm(enumerate(agent_positions), desc=f"Path {path_number}"):
        if folder_counter >= max_folders:
            return folder_counter  # Stop if max folders reached

        # Create a unique folder for each sampled point
        folder_name = generate_folder_name(scene_name, idx)
        point_folder = os.path.join(output_dir, folder_name)
        os.makedirs(point_folder, exist_ok=True)
        folder_counter += 1

        # Set agent state to the current position
        agent_state = habitat_sim.AgentState()
        agent_state.position = position
        agent.set_state(agent_state)

        # Apply predefined rotations to the agent
        for yaw_index, yaw in enumerate(angles):

            agent.agent_config.action_space["turn_right"].actuation.amount = yaw
            agent.act("turn_right")

            # Get observations after the action
            observations = sim.get_sensor_observations()
            rgb = observations["color_sensor"]

            # Save the RGB image
            rgb_filename = os.path.join(point_folder, f"{yaw_index:06d}.png")
            cv2.imwrite(rgb_filename, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

            # Save the pose data (position and quaternion)
            pose_data = np.concatenate((agent.get_state().position, agent.get_state().rotation.components))
            pose_filename = os.path.join(point_folder, f"{yaw_index:06d}.npy")
            np.save(pose_filename, pose_data)

        if folder_counter >= max_folders:
            break  # Stop if max folders reached

    return folder_counter

# Main loop to process multiple scenes
main_directories = [
    "/home/shared/haoze/data/data/scene_datasets/hm3d/train",  # First directory
    "/home/shared/haoze/data/scene_datasets/hm3d/minival",  # First directory
]

# List of json files to exclude
json_files = [
    "hm3d_annotated_minival_basis.scene_dataset_config.json",
    "hm3d_minival_basis.scene_dataset_config.json",
]

max_folders = 15000  # Stop when 15000 folders are created
folder_counter = 0

for main_directory in main_directories:
    for scene_dir in os.listdir(main_directory):
        scene_path = os.path.join(main_directory, scene_dir)

        # Skip json files and non-directories
        if scene_dir in json_files or not os.path.isdir(scene_path):
            continue

        # Extract file prefix based on your directory structure
        # Adjust this part if necessary
        file_prefix = scene_dir.split("-")[-1]
        scene_glb = os.path.join(scene_path, f"{file_prefix}.basis.glb")
        navmesh_path = os.path.join(scene_path, f"{file_prefix}.basis.navmesh")

        if not os.path.exists(scene_glb) or not os.path.exists(navmesh_path):
            print(f"Skipping {scene_dir}: missing .glb or .navmesh file")
            continue

        output_dir = "./indoor_data_partial"
        os.makedirs(output_dir, exist_ok=True)

        # Create simulator configuration
        cfg = make_cfg(scene_glb, 1.5, 512, 512)

        # Create simulator
        sim = habitat_sim.Simulator(cfg)
        sim.recompute_navmesh(sim.pathfinder, habitat_sim.NavMeshSettings())
        sim.pathfinder.load_nav_mesh(navmesh_path)

        # Initialize agent
        agent = sim.initialize_agent(0)

        # Sample two distinct paths
        result = sample_two_different_paths(sim)
        if result is None:
            print(f"Could not find two navigable paths in {scene_dir}. Skipping.")
            sim.close()
            continue
        (start_point_1, end_point_1), (start_point_2, end_point_2), path1, path2 = result

        # Retrieve path points
        path1_points = path1.points
        path2_points = path2.points

        # Get sampled points for both paths
        agent_positions_1 = sample_points_along_path(path1_points, interval=0.5)
        agent_positions_2 = sample_points_along_path(path2_points, interval=0.5)

        # Save data for both paths
        folder_counter = save_data_in_folder(output_dir, scene_dir, agent_positions_1, path_number=1, sim=sim, agent=agent, folder_counter=folder_counter, max_folders=max_folders)
        folder_counter = save_data_in_folder(output_dir, scene_dir, agent_positions_2, path_number=2, sim=sim, agent=agent, folder_counter=folder_counter, max_folders=max_folders)

        if folder_counter >= max_folders:
            print(f"Reached {max_folders} folders. Stopping.")
            sim.close()
            break  # Stop if max folders reached

        # Close simulator
        sim.close()
