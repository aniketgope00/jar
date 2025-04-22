import os
import numpy as np
import open3d as o3d
from PIL import Image
import torch
from transformers import GLPNImageProcessor, GLPNForDepthEstimation

def generate_3d_models(image_path):
    """
    Generate 3D models (PLY, OBJ, GLB) from an input image.

    Parameters:
        image_path (str): Path to the input image.

    Returns:
        None
    """
    # Ensure output directory exists
    output_dir = "GENERATED_3D_MODELS"
    os.makedirs(output_dir, exist_ok=True)

    # Load model and feature extractor
    feature_extractor = GLPNImageProcessor.from_pretrained('vinvino02/glpn-nyu')
    model = GLPNForDepthEstimation.from_pretrained('vinvino02/glpn-nyu')

    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    new_height = 480 if image.height > 480 else image.height
    new_height -= new_height % 32
    new_width = int(new_height * image.width / image.height)
    diff = new_width % 32
    new_width = new_width - diff if diff < 16 else new_width + 32 - diff
    image = image.resize((new_width, new_height))
    inputs = feature_extractor(images=image, return_tensors="pt")

    # Predict depth
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth

    # Final post-processing
    pad = 16
    output = predicted_depth.squeeze().cpu().numpy() * 1000.0
    output = output[pad:-pad, pad:-pad]
    image = image.crop((pad, pad, image.width - pad, image.height - pad))

    # Convert to Open3D RGBD image
    width, height = image.size
    depth_image = (output * 255 / np.max(output)).astype('uint8')
    image = np.array(image)
    depth_o3d = o3d.geometry.Image(depth_image)
    image_o3d = o3d.geometry.Image(image)
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(image_o3d, depth_o3d, convert_rgb_to_intensity=False)

    # Camera projection model
    camera_intrinsic = o3d.camera.PinholeCameraIntrinsic()
    camera_intrinsic.set_intrinsics(width, height, 500, 500, width / 2, height / 2)

    # Create point cloud
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, camera_intrinsic)
    pcd_path = os.path.join(output_dir, "point_cloud.ply")
    o3d.io.write_point_cloud(pcd_path, pcd)

    # Post-process point cloud
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    pcd = pcd.select_by_index(ind)
    pcd.estimate_normals()
    pcd.orient_normals_to_align_with_direction()

    # Surface reconstruction
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=10, n_threads=1)[0]
    rotation = mesh.get_rotation_matrix_from_xyz((np.pi, 0, 0))
    mesh.rotate(rotation, center=(0, 0, 0))

    # Save mesh files
    mesh_glb_path = os.path.join(output_dir, "mesh.glb")
    mesh_obj_path = os.path.join(output_dir, "mesh.obj")
    mesh_ply_path = os.path.join(output_dir, "mesh_rotated.ply")
    o3d.io.write_triangle_mesh(mesh_glb_path, mesh)
    o3d.io.write_triangle_mesh(mesh_obj_path, mesh)
    o3d.io.write_triangle_mesh(mesh_ply_path, mesh)

    # Uniformly paint and save
    mesh_uniform = mesh.paint_uniform_color([0.9, 0.8, 0.9])
    mesh_uniform.compute_vertex_normals()
    mesh_uniform_path = os.path.join(output_dir, "mesh_uniform.obj")
    o3d.io.write_triangle_mesh(mesh_uniform_path, mesh_uniform)

    print(f"3D models saved in {output_dir}")

if __name__ == "__main__":
    generate_3d_models("image_processing_module/truck.jpg")
