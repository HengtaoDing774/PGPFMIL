import h5py
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def visualize_patches_on_slide(h5_path, slide_img_path, w0, h0, patch_size=224,
                               save_path=None, alpha=1.0, color=(211, 129, 121)):
    """
    将 H5 文件中 patch 坐标映射到 WSI 截图上，并可视化。

    参数:
        h5_path: H5 文件路径，包含坐标 dataset "coords"
        slide_img_path: WSI 截图路径（JPEG/PNG）
        w0, h0: WSI 原始大小（Level 0）
        patch_size: patch 原始大小（默认224）
        save_path: 如果指定，保存可视化图片
        alpha: 透明度 (0~1)，默认 1.0 完全不透明
        color: 填充颜色，格式 (R, G, B)
    """

    # 读取 patch 坐标
    with h5py.File(h5_path, "r") as f:
        coords = f["coords"][:]  # shape: (N,2)
    print(f"读取 {coords.shape[0]} 个 patch 坐标")

    # 读取截图
    slide_img = Image.open(slide_img_path)
    slide_w, slide_h = slide_img.size
    print(f"截图尺寸: {slide_w}x{slide_h}")

    # 计算缩放比例
    scale_x = slide_w / w0
    scale_y = slide_h / h0
    print(f"缩放比例: x={scale_x:.4f}, y={scale_y:.4f}")

    # 映射坐标到截图
    mapped_coords = coords.copy()
    mapped_coords[:, 0] = coords[:, 0] * scale_x
    mapped_coords[:, 1] = coords[:, 1] * scale_y

    # 可视化
    plt.figure(figsize=(12, 12))
    plt.imshow(slide_img)
    ax = plt.gca()

    # 转换颜色到 matplotlib [0,1] 范围
    norm_color = tuple([c / 255 for c in color])

    # 绘制统一颜色的矩形
    for (x, y) in mapped_coords:
        rect = Rectangle((x, y), patch_size * scale_x, patch_size * scale_y,
                         linewidth=0.5, edgecolor=norm_color, facecolor=norm_color, alpha=alpha)
        ax.add_patch(rect)

    plt.axis('off')

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"可视化图片已保存: {save_path}")
    plt.show()


# ---------------------------
# 示例调用
h5_path = r"F:\WSI_Code\TransMIL-main\retain_coords_Ours\HE-Y3.h5"
slide_img_path = r"HE-Y3.png"
w0, h0 = 112896, 86016 # WSI 原始大小
patch_size = 224

visualize_patches_on_slide(h5_path, slide_img_path, w0, h0, patch_size,
                           save_path="patch_overlay_fixedcolor17.png",
                           alpha=1.0,
                           color=(211, 129, 121))
