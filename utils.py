import numpy as np
import plyfile
import skimage.measure

import pdb


def N_to_reso(n_voxels, bbox):
    xyz_min, xyz_max = bbox
    dim = len(xyz_min)
    voxel_size = ((xyz_max - xyz_min).prod() / n_voxels).pow(1 / dim)
    return ((xyz_max - xyz_min) / voxel_size).long().tolist()


def cal_n_samples(reso, step_ratio=0.5):
    return int(np.linalg.norm(reso) / step_ratio)


def convert_sdf_samples_to_ply(
    pytorch_3d_sdf_tensor,
    ply_filename_out,
    bbox,
    level=0.5,
):
    """
    Convert sdf samples to .ply

    :param pytorch_3d_sdf_tensor: a torch.FloatTensor of shape (n,n,n)
    :voxel_grid_origin: a list of three floats: the bottom, left, down origin of the voxel grid
    :voxel_size: float, the size of the voxels
    :ply_filename_out: string, path of the filename to save to

    This function adapted from: https://github.com/RobotLocomotion/spartan
    """

    numpy_3d_sdf_tensor = pytorch_3d_sdf_tensor.numpy()
    voxel_size = list((bbox[1] - bbox[0]) / np.array(pytorch_3d_sdf_tensor.shape))

    verts, faces, normals, values = skimage.measure.marching_cubes(numpy_3d_sdf_tensor, level=level, spacing=voxel_size)
    # verts.shape -- (284621, 3)
    # faces.shape -- (572646, 3)
    # normals.shape -- (284621, 3)
    # values.shape -- (284621,)

    faces = faces[..., ::-1]  # inverse face orientation

    # transform from voxel coordinates to camera coordinates
    # note x and y are flipped in the output of marching_cubes
    mesh_points = np.zeros_like(verts)
    mesh_points[:, 0] = bbox[0, 0] + verts[:, 0]
    mesh_points[:, 1] = bbox[0, 1] + verts[:, 1]
    mesh_points[:, 2] = bbox[0, 2] + verts[:, 2]

    # try writing to the ply file
    num_verts = verts.shape[0]
    num_faces = faces.shape[0]

    verts_tuple = np.zeros((num_verts,), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
    for i in range(0, num_verts):
        verts_tuple[i] = tuple(mesh_points[i, :])

    faces_building = []
    for i in range(0, num_faces):
        faces_building.append(((faces[i, :].tolist(),)))
    faces_tuple = np.array(faces_building, dtype=[("vertex_indices", "i4", (3,))])

    el_verts = plyfile.PlyElement.describe(verts_tuple, "vertex")
    el_faces = plyfile.PlyElement.describe(faces_tuple, "face")

    ply_data = plyfile.PlyData([el_verts, el_faces])
    print("saving mesh to %s" % (ply_filename_out))
    ply_data.write(ply_filename_out)
