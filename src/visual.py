# src/visual.py

import matplotlib.pyplot as plt
import numpy as np

def create_comparison_figure(original_slice, original_mask_slice, 
                             resized_slice, resized_mask_slice,
                             original_shape, target_shape, slice_idx, file_name):
    """
    创建一个 2x2 的对比图，用于展示 resize 前后的影像和掩膜。

    只负责生成图形对象，不负责保存或显示。
    """
    # 归一化CT影像以便于可视化
    norm_orig_slice = _normalize_ct_image(original_slice)
    norm_resized_slice = _normalize_ct_image(resized_slice)

    # 创建 2x2 的子图画布
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    fig.suptitle(f'Resize analysis for {file_name}', fontsize=16)

    # 1. 左上: 原图
    axes[0, 0].imshow(np.rot90(norm_orig_slice), cmap='gray')
    axes[0, 0].set_title(f'Original Image (Slice: {slice_idx})\nShape: {original_shape}')
    axes[0, 0].axis('off')
    
    # 2. 右上: 原图 + 原掩膜
    axes[0, 1].imshow(np.rot90(norm_orig_slice), cmap='gray')
    axes[0, 1].imshow(np.rot90(np.ma.masked_where(original_mask_slice == 0, original_mask_slice)), cmap='tab20', alpha=0.5)
    axes[0, 1].set_title('Original Image + Mask')
    axes[0, 1].axis('off')
    
    # 3. 左下: Resize后的图
    axes[1, 0].imshow(np.rot90(norm_resized_slice), cmap='gray')
    axes[1, 0].set_title(f'Resized Image\nShape: {target_shape}')
    axes[1, 0].axis('off')

    # 4. 右下: Resize后的图 + Resize后的掩膜
    axes[1, 1].imshow(np.rot90(norm_resized_slice), cmap='gray')
    axes[1, 1].imshow(np.rot90(np.ma.masked_where(resized_mask_slice == 0, resized_mask_slice)), cmap='tab20', alpha=0.5)
    axes[1, 1].set_title('Resized Image + Resized Mask')
    axes[1, 1].axis('off')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    return fig

def _normalize_ct_image(img_data):
    """内部辅助函数，用于CT影像窗口化。"""
    window_level, window_width = 40, 400
    min_val = window_level - window_width // 2
    max_val = window_level + window_width // 2
    clipped = np.clip(img_data, min_val, max_val)
    # 避免除以零的错误
    if max_val == min_val:
        return np.zeros_like(clipped)
    normalized = (clipped - min_val) / (max_val - min_val)
    return normalized