import numpy as np
import scipy
import time
import cv2
import os
import glob
import math
import trimesh
import objio
import imageio
import pyglet
from mypyrender import IntrinsicsCamera, \
    DirectionalLight, SpotLight, PointLight, \
    MetallicRoughnessMaterial, \
    Primitive, Mesh, Node, Scene, \
    Viewer, OffscreenRenderer


# skin
skin_mat = {
    'color': np.array([[255, 195, 174]]) / 255.0 * 0.8, 
    'metallicFactor': 0.0, 
    'roughnessFactor': 1.0, 
    'ambient_light': 0.5, 
    'directional_light': 1.0, 
    'floor_gray_value': 1.0, 
}

# geometry
geo_mat = {
    'color': np.array([[222, 222, 222]]) / 255.0, 
    'metallicFactor': 0.2, 
    'roughnessFactor': 0.8,     
    'ambient_light': 0.22, 
    'directional_light': 1.8, 
    'floor_gray_value': 1.0, 
}

# color
clr_mat = {
    'color': None, 
    'metallicFactor': None, 
    'roughnessFactor': None,     
    'ambient_light': 0.8, 
    'directional_light': 1.0, 
    'floor_gray_value': 0.78, 
}


def rotationx(theta):
    return np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, np.cos(theta / 180 * np.pi), np.sin(theta / 180 * np.pi), 0.0],
        [0.0, -np.sin(theta / 180 * np.pi), np.cos(theta / 180 * np.pi), 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])


def rotationy(theta):
    return np.array([
        [np.cos(theta / 180 * np.pi), 0.0, np.sin(theta / 180 * np.pi), 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [-np.sin(theta / 180 * np.pi), 0.0, np.cos(theta / 180 * np.pi), 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])


def location(x, y, z):
    return np.array([
        [1.0, 0.0, 0.0, x],
        [0.0, 1.0, 0.0, y],
        [0.0, 0.0, 1.0, z],
        [0.0, 0.0, 0.0, 1.0]
    ])

def main(mesh_path, outdir, view_num, color, metallicFactor=None, roughnessFactor=None, 
         render_shadow=False, ambient_light=0.8, directional_light=1.0, floor_gray_value=0.78, 
         img_res=512, cam_f=5000, cam_c=None):
    os.makedirs(outdir, exist_ok=True)
    scene = Scene(ambient_light=np.array([ambient_light, ambient_light, ambient_light, 1.0]))

    ################
    # set camera
    if cam_c is None:
        cam_c = img_res / 2.0
    cam = IntrinsicsCamera(fx=cam_f, fy=cam_f, cx=cam_c, cy=cam_c)
    cam_pose = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 10.0],
        [0.0, 0.0, 0.0, 1.0]
    ])

    ################
    # set material
    material = None
    if metallicFactor is not None and roughnessFactor is not None:
        material = MetallicRoughnessMaterial(metallicFactor=metallicFactor, roughnessFactor=roughnessFactor)

    ################
    # read mesh
    mesh = objio.load_obj_data(mesh_path)
    if 'vc' not in mesh:
        mesh['vc'] = np.ones_like(mesh['v']) * 0.8
    if color is not None:
        if color.shape[0] == 1:
            mesh['vc'] = np.ones_like(mesh['v']) * color 
        else:
            assert mesh['v'].shape[0] == color.shape[0]
            mesh['vc'] = color
    mesh_ = trimesh.Trimesh(vertices=mesh['v'], faces=mesh['f'], vertex_colors=mesh['vc'])
    points_mesh = Mesh.from_trimesh(mesh_, smooth=True, material=material)
    human_node = scene.add(points_mesh)

    ################
    # add plane
    plane_size = 4
    min_y = np.min(mesh_.vertices[:, 1]) + 0.005
    plane_points = np.array([
        [plane_size, min_y, plane_size],
        [plane_size, min_y, -plane_size],
        [-plane_size, min_y, plane_size],
        [-plane_size, min_y, -plane_size]
    ])
    plane_faces = np.array([
        [0, 1, 2],
        [2, 1, 3]
    ])
    plane_colors = np.ones((4, 3), dtype=np.float32) * floor_gray_value
    plane_mesh = trimesh.Trimesh(
        vertices=plane_points, faces=plane_faces, vertex_colors=plane_colors)
    plane_mesh = Mesh.from_trimesh(plane_mesh, smooth=False)
    plane_node = scene.add(plane_mesh)

    ################
    # add direct lighting
    direc_l = DirectionalLight(color=np.ones(3), intensity=directional_light)
    light_node_1 = scene.add(direc_l, pose=np.matmul(rotationy(30), rotationx(45)))
    direc_l = DirectionalLight(color=np.ones(3), intensity=directional_light)
    light_node_2 = scene.add(direc_l, pose=np.matmul(rotationy(-30), rotationx(45)))
    direc_l = DirectionalLight(color=np.ones(3), intensity=directional_light)
    light_node_3 = scene.add(direc_l, pose=np.matmul(rotationy(-180), rotationx(45)))
    direc_l = DirectionalLight(color=np.ones(3), intensity=(directional_light-0.5))
    light_node_4 = scene.add(direc_l, pose=np.matmul(rotationy(0), rotationx(-10)))

    ################
    # rendering
    cam_node = scene.add(cam, pose=cam_pose)
    render_flags = {
        'flip_wireframe': False,
        'all_wireframe': False,
        'all_solid': False,
        'shadows': render_shadow,
        'vertex_normals': False,
        'face_normals': False,
        'cull_faces': True,
        'point_size': 1.0,
    }
    r = OffscreenRenderer(viewport_width=img_res, viewport_height=img_res, render_flags=render_flags)
    angle = 360 / view_num
    rotmat = rotationy(angle)[:3, :3]
    for view_id in range(view_num):
        mesh['v'] = np.matmul(mesh['v'], rotmat.transpose())
        mesh_ = trimesh.Trimesh(vertices=mesh['v'], faces=mesh['f'], vertex_colors=mesh['vc'])
        points_mesh = Mesh.from_trimesh(mesh_, smooth=True, material=material)
        color, depth = r.render(scene)
        scene.remove_node(human_node)
        human_node = scene.add(points_mesh)
        cv2.imshow('test', color[:, :, ::-1])
        cv2.waitKey(30)

        # # [debugging code]
        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.imshow(color)
        # plt.show()

    r.delete()
    scene.clear()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mesh_path', default=None, type=str)
    parser.add_argument('--out_dir', default=None, type=str)
    parser.add_argument('--material', default=None, type=str)
    parser.add_argument('--shadows', default=False, type=bool)
    args = parser.parse_args()
    if args.material == 'skin':
        main(args.mesh_path, args.out_dir, 60, skin_mat['color'],  
             render_shadow=args.shadows, 
             metallicFactor=skin_mat['metallicFactor'], roughnessFactor=skin_mat['roughnessFactor'], 
             ambient_light=skin_mat['ambient_light'], directional_light=skin_mat['directional_light'], 
             floor_gray_value=skin_mat['floor_gray_value'])
    elif args.material == 'geo':
        main(args.mesh_path, args.out_dir, 60, geo_mat['color'],  
             render_shadow=args.shadows, 
             metallicFactor=geo_mat['metallicFactor'], roughnessFactor=geo_mat['roughnessFactor'], 
             ambient_light=geo_mat['ambient_light'], directional_light=geo_mat['directional_light'], 
             floor_gray_value=skin_mat['floor_gray_value'])
    else:
        main(args.mesh_path, args.out_dir, 60, None, None, None, render_shadow=args.shadows, 
             ambient_light=clr_mat['ambient_light'], directional_light=clr_mat['directional_light'], 
             floor_gray_value=clr_mat['floor_gray_value'])

# Example:
# python .\render_360degree.py --mesh_path .\data\WOMEN_Dresses_id_00005350_03_4_full_tex.obj --out_dir ./debug/ --material geo --shadows True
# python .\render_360degree.py --mesh_path .\data\WOMEN_Dresses_id_00005350_03_4_full_tex.obj --out_dir ./debug/ --material skin --shadows True
# python .\render_360degree.py --mesh_path .\data\WOMEN_Dresses_id_00005350_03_4_full_tex.obj --out_dir ./debug/ --material none --shadows True
