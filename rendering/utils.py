import math
import numpy as np
import cv2
from tqdm import tqdm
import random
import json
from scipy.linalg import expm, norm
import os
from rendering.renderer import Renderer

rows, cols = (480, 640)


def compute_rotation_from_vertex(vertex):
    """Compute rotation matrix from viewpoint vertex """
    up = [0, 0, 1]
    if vertex[0] == 0 and vertex[1] == 0 and vertex[2] != 0:  # 这是在z轴上啊
        up = [-1, 0, 0]  # 在最顶端时
    rot = np.zeros((3, 3))
    rot[:, 2] = -vertex / norm(vertex)  # View direction towards origin
    rot[:, 0] = np.cross(rot[:, 2], up)  # 叉乘 获得相机的右边方向向量
    rot[:, 0] /= norm(rot[:, 0])
    rot[:, 1] = np.cross(rot[:, 0], -rot[:, 2])
    return rot.T


def create_pose(vertex, scale=0, angle_deg=0):
    """Compute transform matrix from viewpoint vertex and inplane rotation """
    rot = compute_rotation_from_vertex(vertex)  # 相机姿态矩阵, 以列向量形式为 [右向量, 上向量, 正前方向量]

    rodriguez = np.asarray([0, 0, 1]) * (angle_deg * math.pi / 180.0)  # 罗德里格斯公式, 模的大小是旋转角度
    # np.cross(np.eye(3), rodriguez)是计算rodriguez的反对称矩阵
    angle_axis = expm(np.cross(np.eye(3), rodriguez))  # 通过指数映射得到旋转矩阵
    # print(rodriguez, angle_axis)
    transform = np.eye(4)
    transform[0:3, 0:3] = np.matmul(angle_axis, rot)  # R ?
    transform[0:3, 3] = [0, 0, scale]  # t ?
    return transform


def project(object3d, Rt, intrinsics):
    Rt = Rt[:3, :]
    cam2d = np.dot(intrinsics, np.dot(Rt, object3d.transpose()))
    cam2d[0, :] = cam2d[0, :] / cam2d[2, :]
    cam2d[1, :] = cam2d[1, :] / cam2d[2, :]
    cam2d = cam2d.astype(np.int32).transpose((1, 0))
    return cam2d


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def precompute_projections(dataset, views, cam, model_map, models):
    """Precomputes the projection information needed for 6D pose construction
    """
    save_path = f"output/{dataset}"  # save the rendered images
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    w, h = 640, 480

    ren = Renderer((w, h), cam)
    for model_name in models:
        count = 0  # 图片计数
        scene_camera = {}
        scene_gt = {}
        model_path = os.path.join(save_path, model_name[-6:])
        rgb_dir = os.path.join(model_path, 'rgb')
        depth_dir = os.path.join(model_path, 'mask')
        ensure_dir(rgb_dir)
        ensure_dir(depth_dir)

        for i in tqdm(range(len(views))):
            pose = create_pose(views[i], angle_deg=0)
            pose[:3, 3] = [0, 0, 0.5]  # zr = 0.5
            model = model_map[model_name]  # 随机选出一个模型
            ren.clear()
            ren.draw_model(model, pose)
            # ren.draw_boundingbox(model, pose)
            col, dep = ren.finish()  # dep 缩放因子是0.001
            col *= 255
            dep[dep > 0] = 255
            # cam2d = project(model.fps, pose, cam)
            # ys, xs = np.nonzero(dep > 0)
            # xs_min = xs_original.min()
            # ys_min = ys_original.min()
            # xs_max = xs_original.max()
            # ys_max = ys_original.max()
            scene_camera[str(i)] = {
                'cam_K': cam.flatten().tolist(),
                'depth_scale': 1,
                'view_level': 4,
            }

            scene_gt[str(i)] = [{
                'cam_R_m2c': pose[:3, :3].flatten().tolist(),
                'cam_t_m2c': pose[:3, 3].flatten().tolist(),
                'obj_id': int(model_name[-2:])
            }]
            # draw fps
            # for i_p, point in enumerate(cam2d):
            #     if i_p == 0:
            #         col = cv2.circle(col, (point[0], point[1]), 3, (0, 255, 0), thickness=-1)
            #     else:
            #         col = cv2.circle(col, (point[0], point[1]), 3, (0, 0, 255), thickness=-1)
            cv2.imwrite(os.path.join(rgb_dir, str(count).zfill(6) + '.png'), col)
            cv2.imwrite(os.path.join(depth_dir, str(count).zfill(6) + '_000000' + '.png'), dep)
            count += 1
        save_json(os.path.join(model_path, 'scene_camera.json'), scene_camera)
        save_json(os.path.join(model_path, 'scene_gt.json'), scene_gt)
        # json.dump(scene_gt, )


def save_json(path, content):
    with open(path, 'w') as f:
        if isinstance(content, dict):
            f.write('{\n')
            content_sorted = sorted(content.items(), key=lambda x: int(x[0]))
            for elem_id, (k, v) in enumerate(content_sorted):
                f.write('  \"{}\": {}'.format(k, json.dumps(v, sort_keys=True)))
                if elem_id != len(content) - 1:
                    f.write(',')
                f.write('\n')
            f.write('}')

        elif isinstance(content, list):
            f.write('[\n')
            for elem_id, elem in enumerate(content):
                f.write('  {}'.format(json.dumps(elem, sort_keys=True)))
                if elem_id != len(content) - 1:
                    f.write(',')
                f.write('\n')
            f.write(']')

        else:
            json.dump(content, f, sort_keys=True)