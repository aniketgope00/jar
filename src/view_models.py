from open3d import io, visualization  # Import Open3D for point cloud and mesh handling
import open3d as o3d  # Ensure open3d is imported as o3d
import numpy as np
import os

import trimesh


def load_ply(filename):
    """
    Loads a .ply file and extracts vertices and faces.
    
    Args:
        filename (str): Path to the .ply file.
    
    Returns:
        tuple: A tuple containing vertices (numpy.ndarray) and faces (list of lists).
    """
    with open(filename, 'r', encoding='utf-8') as f:  # Specify UTF-8 encoding
        lines = f.readlines()

    vertices = []
    faces = []

    in_header = True
    for line in lines:
        if in_header:
            if line.startswith("end_header"):
                in_header = False
            continue

        parts = line.strip().split()
        if len(parts) == 3:  # vertex
            vertices.append([float(p) for p in parts])
        elif len(parts) > 3:  # face
            count = int(parts[0])
            face = [int(i) for i in parts[1:1+count]]
            faces.append(face)

    return np.array(vertices, dtype='f'), faces


def show_ply_with_open3d(filepath):
    """
    Displays a .ply model using Open3D's visualization.
    
    Args:
        filepath (str): Path to the .ply file.
    """
    cloud = io.read_point_cloud(filepath)  # Read point cloud
    visualization.draw_geometries([cloud])  # Visualize point cloud


def show_obj_with_open3d(filepath):
    """
    Displays an .obj model using Open3D's visualization.
    
    Args:
        filepath (str): Path to the .obj file.
    """
    mesh = io.read_triangle_mesh(filepath)  # Read the .obj file as a mesh
    if not mesh.has_vertices():
        print(f"File {filepath} does not contain valid geometry.")
        return
    visualization.draw_geometries([mesh])  # Visualize the mesh


def show_glb(filepath):
    """
    Displays a .glb model using Open3D's visualization.

    Args:
        filepath (str): Path to the .glb file.
    """
    import trimesh  # Ensure trimesh is imported locally
    mesh = trimesh.load(filepath, force='mesh')
    if not isinstance(mesh, trimesh.Trimesh):
        print("Loaded object is not a Trimesh.")
        return

    # Convert Trimesh to Open3D
    o3d_mesh = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(mesh.vertices),
        triangles=o3d.utility.Vector3iVector(mesh.faces)
    )
    o3d.visualization.draw_geometries([o3d_mesh])


if __name__ == "__main__":
    # Path to the .glb file
    glb_path = "GENERATED_3D_MODELS/mesh.glb"
    if os.path.exists(glb_path):
        # Display the .glb model with Open3D visualization
        show_glb(glb_path)
    else:
        print(f"File {glb_path} does not exist.")

    ''' 
    # Path to the .obj file
    obj_path = "WORKING_IMAGE/mesh_rotated.obj"
    if os.path.exists(obj_path):
        # Display the .obj model with Open3D visualization
        show_obj_with_open3d(obj_path)
    else:
        print(f"File {obj_path} does not exist.")

    # Path to the .ply file
    ply_path = "WORKING_IMAGE/point_cloud.ply"
    if os.path.exists(ply_path):
        # Display the .ply model with Open3D visualization
        show_ply_with_open3d(ply_path)
    else:
        print(f"File {ply_path} does not exist.")
    '''