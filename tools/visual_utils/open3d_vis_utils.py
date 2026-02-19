"""
Open3d visualization tool box
Written by Jihan YANG
All rights preserved from 2021 - present.
"""
import os
import open3d
import torch
import matplotlib
import numpy as np

box_colormap = [
    [1, 1, 1],
    [0, 1, 0],
    [0, 1, 1],
    [1, 1, 0],
]


def get_coor_colors(obj_labels):
    """
    Args:
        obj_labels: 1 is ground, labels > 1 indicates different instance cluster

    Returns:
        rgb: [N, 3]. color for each point.
    """
    colors = matplotlib.colors.XKCD_COLORS.values()
    max_color_num = obj_labels.max()

    color_list = list(colors)[:max_color_num+1]
    colors_rgba = [matplotlib.colors.to_rgba_array(color) for color in color_list]
    label_rgba = np.array(colors_rgba)[obj_labels]
    label_rgba = label_rgba.squeeze()[:, :3]

    return label_rgba


def draw_scenes(points, gt_boxes=None, ref_boxes=None, ref_labels=None, ref_scores=None, point_colors=None, draw_origin=True, save_path=None):
    if isinstance(points, torch.Tensor):
        points = points.cpu().numpy()
    if isinstance(gt_boxes, torch.Tensor):
        gt_boxes = gt_boxes.cpu().numpy()
    if isinstance(ref_boxes, torch.Tensor):
        ref_boxes = ref_boxes.cpu().numpy()

    if save_path is not None:
        try:
            _draw_offscreen(points, gt_boxes, ref_boxes, ref_labels, ref_scores, point_colors, draw_origin, save_path)
        except Exception as e:
            print(f'[Headless] OffscreenRenderer 실패: {e}')
            print('[Headless] matplotlib BEV 이미지로 저장합니다.')
            _draw_bev(points, gt_boxes, ref_boxes, ref_labels, ref_scores, save_path)
        return

    vis = open3d.visualization.Visualizer()
    vis.create_window()

    vis.get_render_option().point_size = 1.0
    vis.get_render_option().background_color = np.zeros(3)

    # draw origin
    if draw_origin:
        axis_pcd = open3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
        vis.add_geometry(axis_pcd)

    pts = open3d.geometry.PointCloud()
    pts.points = open3d.utility.Vector3dVector(points[:, :3])

    vis.add_geometry(pts)
    if point_colors is None:
        pts.colors = open3d.utility.Vector3dVector(np.ones((points.shape[0], 3)))
    else:
        pts.colors = open3d.utility.Vector3dVector(point_colors)

    if gt_boxes is not None:
        vis = draw_box(vis, gt_boxes, (0, 0, 1))

    if ref_boxes is not None:
        vis = draw_box(vis, ref_boxes, (0, 1, 0), ref_labels, ref_scores)

    vis.run()
    vis.destroy_window()


def _draw_offscreen(points, gt_boxes, ref_boxes, ref_labels, ref_scores, point_colors, draw_origin, save_path):
    """OffscreenRenderer를 사용한 헤드리스 렌더링 (EGL 필요)"""
    import open3d.visualization.rendering as rendering

    renderer = rendering.OffscreenRenderer(1920, 1080)
    renderer.scene.set_background([0.0, 0.0, 0.0, 1.0])

    mat_pts = rendering.MaterialRecord()
    mat_pts.shader = 'defaultUnlit'
    mat_pts.point_size = 2.0

    pts = open3d.geometry.PointCloud()
    pts.points = open3d.utility.Vector3dVector(points[:, :3])
    if point_colors is None:
        pts.colors = open3d.utility.Vector3dVector(np.ones((points.shape[0], 3)))
    else:
        pts.colors = open3d.utility.Vector3dVector(point_colors)
    renderer.scene.add_geometry('points', pts, mat_pts)

    if draw_origin:
        axis_pcd = open3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
        mat_axis = rendering.MaterialRecord()
        mat_axis.shader = 'defaultLit'
        renderer.scene.add_geometry('axis', axis_pcd, mat_axis)

    mat_line = rendering.MaterialRecord()
    mat_line.shader = 'unlitLine'
    mat_line.line_width = 2.0

    if gt_boxes is not None:
        for i in range(gt_boxes.shape[0]):
            line_set, _ = translate_boxes_to_open3d_instance(gt_boxes[i])
            line_set.paint_uniform_color([0, 0, 1])
            renderer.scene.add_geometry(f'gt_{i}', line_set, mat_line)

    if ref_boxes is not None:
        for i in range(ref_boxes.shape[0]):
            line_set, _ = translate_boxes_to_open3d_instance(ref_boxes[i])
            color = box_colormap[ref_labels[i] % len(box_colormap)] if ref_labels is not None else [0, 1, 0]
            line_set.paint_uniform_color(color)
            renderer.scene.add_geometry(f'ref_{i}', line_set, mat_line)

    bounds = renderer.scene.bounding_box
    center = bounds.get_center()
    eye = center + np.array([0.0, -60.0, 60.0])
    renderer.setup_camera(60.0, center.tolist(), eye.tolist(), [0.0, 0.0, 1.0])

    img = renderer.render_to_image()
    open3d.io.write_image(save_path, img)
    print(f'[Headless] 3D 결과 이미지 저장: {save_path}')


def _draw_bev(points, gt_boxes, ref_boxes, ref_labels, ref_scores, save_path):
    """matplotlib을 이용한 BEV(Bird's Eye View) 이미지 저장"""
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    color_names = ['white', 'lime', 'cyan', 'yellow']

    fig, ax = plt.subplots(figsize=(20, 20))
    ax.set_facecolor('#111111')
    fig.patch.set_facecolor('#111111')

    # 포인트 클라우드 BEV 표시 (x=전방, y=좌측)
    mask = (points[:, 0] > -60) & (points[:, 0] < 80) & (np.abs(points[:, 1]) < 60)
    ax.scatter(points[mask, 1], points[mask, 0], c='lightgray', s=0.3, alpha=0.4)

    def draw_bev_box(box, color, score=None):
        x, y, z, l, w, h, yaw = box[:7]
        cos_a, sin_a = np.cos(-yaw), np.sin(-yaw)
        corners = np.array([
            [ l/2,  w/2], [ l/2, -w/2],
            [-l/2, -w/2], [-l/2,  w/2], [ l/2,  w/2]
        ])
        rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        c = corners @ rot.T
        ax.plot(c[:, 1] + y, c[:, 0] + x, c=color, linewidth=1.5)
        # 전방 방향 표시
        front = np.array([[l/2, 0]]) @ rot.T
        ax.annotate('', xy=(front[0, 1] + y, front[0, 0] + x), xytext=(y, x),
                    arrowprops=dict(arrowstyle='->', color=color, lw=1.5))
        if score is not None:
            ax.text(y, x, f'{score:.2f}', color=color, fontsize=6, ha='center')

    if gt_boxes is not None:
        for box in gt_boxes:
            draw_bev_box(box, 'dodgerblue')

    if ref_boxes is not None:
        for i, box in enumerate(ref_boxes):
            color = color_names[ref_labels[i] % len(color_names)] if ref_labels is not None else 'lime'
            score = float(ref_scores[i]) if ref_scores is not None else None
            draw_bev_box(box, color, score)

    ax.set_xlim(-50, 50)
    ax.set_ylim(-10, 80)
    ax.set_aspect('equal')
    ax.set_xlabel('Y (m)', color='white', fontsize=12)
    ax.set_ylabel('X forward (m)', color='white', fontsize=12)
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_color('gray')
    ax.set_title('BEV Detection Result', color='white', fontsize=14)
    ax.plot(0, 0, 'r+', markersize=12, markeredgewidth=2)  # 센서 위치

    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight', facecolor='#111111')
    plt.close()
    print(f'[Headless] BEV 이미지 저장: {save_path}')


def translate_boxes_to_open3d_instance(gt_boxes):
    """
             4-------- 6
           /|         /|
          5 -------- 3 .
          | |        | |
          . 7 -------- 1
          |/         |/
          2 -------- 0
    """
    center = gt_boxes[0:3]
    lwh = gt_boxes[3:6]
    axis_angles = np.array([0, 0, gt_boxes[6] + 1e-10])
    rot = open3d.geometry.get_rotation_matrix_from_axis_angle(axis_angles)
    box3d = open3d.geometry.OrientedBoundingBox(center, rot, lwh)

    line_set = open3d.geometry.LineSet.create_from_oriented_bounding_box(box3d)

    # import ipdb; ipdb.set_trace(context=20)
    lines = np.asarray(line_set.lines)
    lines = np.concatenate([lines, np.array([[1, 4], [7, 6]])], axis=0)

    line_set.lines = open3d.utility.Vector2iVector(lines)

    return line_set, box3d


def draw_box(vis, gt_boxes, color=(0, 1, 0), ref_labels=None, score=None):
    for i in range(gt_boxes.shape[0]):
        line_set, box3d = translate_boxes_to_open3d_instance(gt_boxes[i])
        if ref_labels is None:
            line_set.paint_uniform_color(color)
        else:
            line_set.paint_uniform_color(box_colormap[ref_labels[i]])

        vis.add_geometry(line_set)

        # if score is not None:
        #     corners = box3d.get_box_points()
        #     vis.add_3d_label(corners[5], '%.2f' % score[i])
    return vis
