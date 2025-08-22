import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
import torch
from scipy.ndimage import zoom
from scipy.ndimage import find_objects

from visual import create_comparison_figure
from data import load_data

import monai.transforms as mtf

# 一些常量。注意自定义。
# 懒得用参数了直接写成硬编码了QAQ
BASE_DIR = "/home/fpf/Med_LLM"
DATA_BASE_DIR = os.path.join(BASE_DIR, "data", "CT_RATE", "dataset")

OUTPUT_DIR = os.path.join(BASE_DIR, "processing_output_pic")
os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET_SHAPE_3D = (384, 384, 32)



# 之前是简单zoom三次插值。换用monai库，一个更加成熟的实现。
# 但是注意维度排序。。。而且必须是4维。下面有点混乱了。一堆维度转换。
def resize_image(image, transform_pipeline):
    print("信息：MONAI 执行 resize_image()。")
    image_with_channel = np.expand_dims(image, axis=0)
    image_monai_format = np.transpose(image_with_channel, (0, 3, 2, 1))

    image_tensor = torch.from_numpy(image_monai_format.copy())

    # transform pipeline
    resized_tensor = transform_pipeline(image_tensor)

    # Tensor 转回 Numpy 数组
    resized_numpy = resized_tensor.numpy()

    #  (C, D, H, W) -> (W, H, D)
    # 变回 (1, W, H, D)
    image_original_format = np.transpose(resized_numpy, (0, 3, 2, 1))
    # 去 C 维度，变回 (W, H, D)
    final_image = np.squeeze(image_original_format, axis=0)

    return final_image
    
def resize_mask(mask, transform_pipeline):

    if not np.issubdtype(mask.dtype, np.integer):
        mask = mask.astype(np.int32)

    organ_ids = np.unique(mask)
    organ_ids = organ_ids[organ_ids != 0]

    monai_target_size = transform_pipeline.transforms[0].spatial_size
    target_shape = (monai_target_size[2], monai_target_size[1], monai_target_size[0])

    if len(organ_ids) == 0:
        return np.zeros(target_shape, dtype=mask.dtype)
    
    # 创建4D概率volume，用于存储所有器官的概率图
    prob_volume = np.zeros((organ_ids.max() + 1, *target_shape), dtype=np.float32)

    # 对每一个器官，在完整的、未裁剪的掩膜上进行操作
    for organ_id in organ_ids:
        # 创建一个与原始掩膜尺寸完全相同的全尺寸二值掩膜
        binary_mask_full_size = (mask == organ_id).astype(np.float32)

        # 将全尺寸二值掩膜传递给 MONAI 进行缩放
        # 转换为 (C, D, H, W)
        binary_mask_with_channel = np.expand_dims(binary_mask_full_size, axis=0)
        binary_mask_monai_format = np.transpose(binary_mask_with_channel, (0, 3, 2, 1))
        binary_tensor = torch.from_numpy(binary_mask_monai_format.copy())

        # 一样变换
        resized_tensor = transform_pipeline(binary_tensor)
        
        # 转回numpy格式
        resized_numpy = resized_tensor.numpy()
        resized_prob_map = np.squeeze(np.transpose(resized_numpy, (0, 3, 2, 1)), axis=0)
        
        # 直接放入概率volume的对应层
        prob_volume[organ_id] = resized_prob_map

    # 决策
    final_mask = np.argmax(prob_volume, axis=0)
    max_probs = np.max(prob_volume, axis=0)
    final_mask[max_probs < 0.3] = 0
    
    return final_mask.astype(mask.dtype)


def parse_arguments():
    parser = argparse.ArgumentParser(description="处理并可视化3D医疗影像。")
    parser.add_argument('--visualize', action='store_true', 
                        help="如果设置此项，将在屏幕上显示可视化结果。")

    return parser.parse_args()


def main():
    args = parse_arguments()

    # 这里未来打算做一个扫描 + 遍历，subid只有1和2。
    #目前只做了单图片的选择。应该，很好改的。
    test_img_id = '1'
    test_sub_id = '1'

    monai_target_size = [TARGET_SHAPE_3D[2], TARGET_SHAPE_3D[1], TARGET_SHAPE_3D[0]]

    monai_transform = mtf.Compose([
        mtf.Resize(spatial_size=monai_target_size, mode="bilinear", align_corners=False)
    ])

    data_pack = load_data(DATA_BASE_DIR, test_img_id, test_sub_id, verbose=False) # 默认简洁输出

    if data_pack:
        original_image = data_pack["image_data"]
        original_mask = data_pack["mask_data"]
        file_name = data_pack["file_name"]
        
        original_shape_3d = original_image.shape
        target_shape_3d = TARGET_SHAPE_3D
        print(f"处理文件: {file_name}")
        print(f"原始尺寸: {original_shape_3d} -> 目标尺寸: {target_shape_3d}")
        
        resized_image = resize_image(original_image, monai_transform)
        resized_mask = resize_mask(original_mask, monai_transform)
        
        # 准备可视化所需的数据切片
        slice_idx_original = original_shape_3d[2] // 2
        slice_idx_resized = target_shape_3d[2] // 2 # 使用新的目标深度

        original_slice = original_image[:, :, slice_idx_original]
        original_mask_slice = original_mask[:, :, slice_idx_original]
        resized_slice = resized_image[:, :, slice_idx_resized]
        resized_mask_slice = resized_mask[:, :, slice_idx_resized]

        comparison_figure = create_comparison_figure(
            original_slice, original_mask_slice,
            resized_slice, resized_mask_slice,
            original_shape_3d, target_shape_3d, slice_idx_original, file_name
        )

        # 保存逻辑
        output_filename = os.path.join(OUTPUT_DIR, f"resize_comparison_{file_name.replace('.nii.gz', '')}.png")
        comparison_figure.savefig(output_filename)
        print(f"对比图已保存至: {output_filename}")

        # 显示逻辑
        if args.visualize:
            print("参数 '--visualize' 已设置，正在显示结果窗口...")
            plt.show()
        
        plt.close(comparison_figure)

        print("处理完成。")

if __name__ == "__main__":
    main()