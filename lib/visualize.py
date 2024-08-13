import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from torchvision.utils import save_image
# import im2mesh.common as common
import numpy as np
import k3d
from matplotlib import cm, colors
import trimesh
from pathlib import Path


def visualize_data(data, data_type, out_file):
    r''' Visualizes the data with regard to its type.
    Args:
        data (tensor): batch of data
        data_type (string): data type (img, voxels or pointcloud)
        out_file (string): output file
    '''
    if data_type == 'img':
        if data.dim() == 3:
            data = data.unsqueeze(0)
        save_image(data, out_file, nrow=4)
    elif data_type == 'voxels':
        visualize_voxels(data, out_file=out_file)
    elif data_type == 'pointcloud':
        visualize_pointcloud(data, out_file=out_file)
    elif data_type is None or data_type == 'idx':
        pass
    else:
        raise ValueError('Invalid data_type "%s"' % data_type)


def visualize_voxels(voxels, out_file=None, show=False):
    r''' Visualizes voxel data.
    Args:
        voxels (tensor): voxel data
        out_file (string): output file
        show (bool): whether the plot should be shown
    '''
    # Use numpy
    voxels = np.asarray(voxels)
    # Create plot
    fig = plt.figure()
    ax = fig.add_subplot(projection=Axes3D.name)
    voxels = voxels.transpose(2, 0, 1)
    ax.voxels(voxels, edgecolor='k')
    ax.set_xlabel('Z')
    ax.set_ylabel('X')
    ax.set_zlabel('Y')
    ax.view_init(elev=30, azim=45)
    if out_file is not None:
        plt.savefig(out_file)
    if show:
        plt.show()
    plt.close(fig)


def visualize_pointcloud(points, normals=None,
                         out_file=None, show=False):
    r''' Visualizes point cloud data.
    Args:
        points (tensor): point data
        normals (tensor): normal data (if existing)
        out_file (string): output file
        show (bool): whether the plot should be shown
    '''
    # Use numpy
    points = np.asarray(points)
    # Create plot
    fig = plt.figure()
    ax = fig.add_subplot(projection=Axes3D.name)
    ax.scatter(points[:, 2], points[:, 0], points[:, 1])
    if normals is not None:
        ax.quiver(
            points[:, 2], points[:, 0], points[:, 1],
            normals[:, 2], normals[:, 0], normals[:, 1],
            length=0.1, color='k'
        )
    ax.set_xlabel('Z')
    ax.set_ylabel('X')
    ax.set_zlabel('Y')
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(-0.5, 0.5)
    ax.set_zlim(-0.5, 0.5)
    ax.view_init(elev=30, azim=45)
    if out_file is not None:
        plt.savefig(out_file)
    if show:
        plt.show()
    plt.close(fig)

def visualise_projection(
        self, points, world_mat, camera_mat, img, output_file='out.png'):
    r''' Visualizes the transformation and projection to image plane.
        The first points of the batch are transformed and projected to the
        respective image. After performing the relevant transformations, the
        visualization is saved in the provided output_file path.
    Arguments:
        points (tensor): batch of point cloud points
        world_mat (tensor): batch of matrices to rotate pc to camera-based
                coordinates
        camera_mat (tensor): batch of camera matrices to project to 2D image
                plane
        img (tensor): tensor of batch GT image files
        output_file (string): where the output should be saved
    '''
    points_transformed = common.transform_points(points, world_mat)
    points_img = common.project_to_camera(points_transformed, camera_mat)
    pimg2 = points_img[0].detach().cpu().numpy()
    image = img[0].cpu().numpy()
    plt.imshow(image.transpose(1, 2, 0))
    plt.plot(
        (pimg2[:, 0] + 1)*image.shape[1]/2,
        (pimg2[:, 1] + 1) * image.shape[2]/2, 'x')
    plt.savefig(output_file)

def visualize_mesh(vertices, faces, file_name, flip_axes=False):
    vertices = np.array(vertices)
    plot = k3d.plot(name='mesh', grid_visible=False, grid=(-0.55, -0.55, -0.55, 0.55, 0.55, 0.55))
    if flip_axes:
        rot_matrix = np.array([
            [-1.0000000, 0.0000000, 0.0000000],
            [0.0000000, 0.0000000, 1.0000000],
            [0.0000000, 1.0000000, 0.0000000]
        ])
        vertices = vertices @ rot_matrix
    plt_mesh = k3d.mesh(vertices.astype(np.float32), faces.astype(np.uint32), color=0xd0d0d0)
    plot += plt_mesh
    plt_mesh.shader = '3d'
    plot.display()


def visualize_meshes(meshes, flip_axes=False):
    assert len(meshes) == 3
    plot = k3d.plot(name='meshes', grid_visible=False, grid=(-0.55, -0.55, -0.55, 0.55, 0.55, 0.55))
    for mesh_idx, mesh in enumerate(meshes):
        vertices, faces = mesh[:2]
        if flip_axes:
            vertices[:, 2] = vertices[:, 2] * -1
            vertices[:, [0, 1, 2]] = vertices[:, [0, 2, 1]]
        vertices += [[-32, -32, 0], [0, -32, 0], [32, -32, 0]][mesh_idx]
        plt_mesh = k3d.mesh(vertices.astype(np.float32), faces.astype(np.uint32), color=0xd0d0d0)
        plot += plt_mesh
        plt_mesh.shader = '3d'
    plot.display()


def visualize_sdf(sdf: np.array, filename: Path) -> None:
    assert sdf.shape[0] == sdf.shape[1] == sdf.shape[2], "SDF grid has to be of cubic shape"
    print(f"Creating SDF visualization for {sdf.shape[0]}^3 grid ...")

    voxels = np.stack(np.meshgrid(range(sdf.shape[0]), range(sdf.shape[1]), range(sdf.shape[2]))).reshape(3, -1).T

    sdf[sdf < 0] /= np.abs(sdf[sdf < 0]).max()
    sdf[sdf > 0] /= sdf[sdf > 0].max()
    sdf /= 2.

    corners = np.array([
        [-.25, -.25, -.25],
        [.25, -.25, -.25],
        [-.25, .25, -.25],
        [.25, .25, -.25],
        [-.25, -.25, .25],
        [.25, -.25, .25],
        [-.25, .25, .25],
        [.25, .25, .25]
    ])[np.newaxis, :].repeat(voxels.shape[0], axis=0).reshape(-1, 3)

    scale_factors = sdf[tuple(voxels.T)].repeat(8, axis=0)
    cube_vertex_colors = cm.get_cmap('seismic')(colors.Normalize(vmin=-1, vmax=1)(scale_factors))[:, :3]
    scale_factors[scale_factors < 0] *= .25
    cube_vertices = voxels.repeat(8, axis=0) + corners * scale_factors[:, np.newaxis]

    faces = np.array([
        [1, 0, 2], [2, 3, 1], [5, 1, 3], [3, 7, 5], [4, 5, 7], [7, 6, 4],
        [0, 4, 6], [6, 2, 0], [3, 2, 6], [6, 7, 3], [5, 4, 0], [0, 1, 5]
    ])[np.newaxis, :].repeat(voxels.shape[0], axis=0).reshape(-1, 3)
    cube_faces = faces + (np.arange(0, voxels.shape[0]) * 8)[np.newaxis, :].repeat(12, axis=0).T.flatten()[:, np.newaxis]

    mesh = trimesh.Trimesh(vertices=cube_vertices, faces=cube_faces, vertex_colors=cube_vertex_colors, process=False)
    mesh.export(str(filename))
    print(f"Exported to {filename}")