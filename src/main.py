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
# 懒得用参数了直接写成硬编码了。
BASE_DIR = "/home/fpf/Med_LLM"
DATA_BASE_DIR = os.path.join(BASE_DIR, "data", "CT_RATE", "dataset")

OUTPUT_DIR = os.path.join(BASE_DIR, "processing_output_pic")
os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET_SHAPE_3D = (384, 384, 32)

'''
def resize_image(image, target_shape):
    input:
    1. image: 图像源文件。
    2. target_shape：想要resize的尺寸
    
    output:
    1. resized_image：resize后的图像
'''

'''
    original_shape = image.shape

    zoom_factors = (
        target_shape[0] / original_shape[0], # X轴（宽度）
        target_shape[1] / original_shape[1], # Y轴（高度）
        target_shape[2] / original_shape[2]  # Z轴（深度）的target和original相同，所以因子为1
    )

    # 直接用三次样条插值。效果不好再改。
    resized_image = zoom(image, zoom_factors, order=3, mode='nearest')

    return resized_image
'''

    # 接下来换用monai库，一个更加成熟的实现。
    # 但是注意维度排序。。。而且必须是4维
def resize_image(image, transform_pipeline):
    print("信息：MONAI 执行 resize_image()。")
    image_with_channel = np.expand_dims(image, axis=0)
    image_monai_format = np.transpose(image_with_channel, (0, 3, 2, 1))

    image_tensor = torch.from_numpy(image_monai_format.copy()) # .copy() 避免 stride 问题

    # 3. 应用 MONAI 的 transform pipeline
    resized_tensor = transform_pipeline(image_tensor)

    # 4. 将处理后的 Tensor 转回 Numpy 数组
    resized_numpy = resized_tensor.numpy()

    # 5. 维度重排：从 (C, D, H, W) -> (W, H, D)
    # a. 交换维度，变回 (1, W, H, D)
    image_original_format = np.transpose(resized_numpy, (0, 3, 2, 1))
    # b. 去掉 Channel 维度，变回 (W, H, D)
    final_image = np.squeeze(image_original_format, axis=0)

    return final_image

# def resize_mask(mask, target_shape):
    # print("信息：正在使用高级语义保持算法执行 resize_mask()。")
    
    # original_shape = mask.shape
    
    # # 第一步：产生新掩膜
    # # 获取除背景(0)以外的所有唯一器官ID
    # organ_ids = np.unique(mask)
    # # 你确定空掩膜的ID是0吗？
    # organ_ids = organ_ids[organ_ids != 0]

    # # 如果掩膜中没有器官（只有背景），直接返回一个全零的目标尺寸掩膜
    # # 否则，进行下一步处理
    # if len(organ_ids) == 0:
    #     return np.zeros(target_shape, dtype=mask.dtype)
        
    # # 计算缩放因子
    # zoom_factors = (
    #     target_shape[0] / original_shape[0],
    #     target_shape[1] / original_shape[1],
    #     target_shape[2] / original_shape[2]
    # )
    
    # # 用于存储每个器官的缩放后概率图的列表
    # resized_prob_maps = []

    # # 第二步：实现缩放

    # for organ_id in organ_ids:
    #     # 为当前器官创建一个二值掩膜 (1.0 vs 0.0)
    #     binary_mask = (mask == organ_id).astype(np.float32)
        
    #     # 三次插值算法进行平滑缩放
    #     resized_prob_map = zoom(binary_mask, zoom_factors, order=3, mode='nearest')
        
    #     # 将缩放后的概率图存入列表
    #     resized_prob_maps.append(resized_prob_map)

    # # 第三步：重组

    # # a. 将所有概率图沿着一个新的轴（axis=0）堆叠起来
    # # 结果是一个4D数组，形状为 (器官数量, width, height, depth)
    # stacked_maps = np.stack(resized_prob_maps, axis=0)

    # # b. 找到每个像素点上概率最大的那个器官的“索引”
    # # argmax 的结果是一个3D数组，其值是 0, 1, 2... 代表在堆叠数组中的索引
    # argmax_indices = np.argmax(stacked_maps, axis=0)
    
    # # c. 将这些索引映射回原始的器官ID
    # # organ_ids 是一个像 [1, 5, 8] 这样的数组
    # # 如果 argmax_indices 在某点的值是 1, 那么 organ_ids[1] 就是 5
    # # 这就完成了从索引到真实ID的转换
    # final_mask = organ_ids[argmax_indices]
    
    # max_probs = np.max(stacked_maps, axis=0)
    # background_pixels = max_probs < 0.5
    # final_mask[background_pixels] = 0

    # # 返回与原始掩膜数据类型相同的最终掩膜
    # return final_mask.astype(mask.dtype)



    # 以下这部分是使用最近邻插值的方法
    '''
    print("信息：最近邻执行 resize_mask()。")
    
    original_shape = mask.shape
    zoom_factors = (
        target_shape[0] / original_shape[0],
        target_shape[1] / original_shape[1],
        target_shape[2] / original_shape[2]
    )
    
    # 使用 order=0 (最近邻插值) 来处理标签掩膜，速度极快
    resized_mask = zoom(mask, zoom_factors, order=0, mode='nearest')
    
    return resized_mask.astype(mask.dtype)
    '''

    # 以下是采用bounding box约束计算量
    '''
    from scipy.ndimage import find_objects

    print("信息：新算法执行 resize_mask()。")

    # --- 关键修复 ---
    # find_objects 函数要求输入一个整数类型的数组。
    # 我们的 .nii.gz 文件将掩膜加载为了 float64，所以必须在这里进行转换。
    if not np.issubdtype(mask.dtype, np.integer):
        mask = mask.astype(np.int32)

    original_shape = np.array(mask.shape)
    target_shape = np.array(target_shape)
    zoom_factors = target_shape / original_shape

    # 找到所有非背景物体的边界框。
    # locations 的索引是 label - 1。例如，标签1的边界框是 locations[0]。
    locations = find_objects(mask)
    
    # 获取掩膜中实际存在的所有器官ID
    organ_ids = np.unique(mask)
    organ_ids = organ_ids[organ_ids != 0]

    # 创建一个4D的概率volume来存储所有器官的概率图，避免复杂的覆盖逻辑
    # 它的形状是 (最大ID+1, width, height, depth)
    prob_volume = np.zeros((organ_ids.max() + 1, *target_shape), dtype=np.float32)

    # 遍历每个实际存在的器官ID
    for organ_id in organ_ids:
        # 从 locations 列表中获取对应ID的边界框 (slice对象)
        # 索引是 organ_id - 1
        loc = locations[organ_id - 1]
        
        # 1. 裁剪出仅包含当前器官及其周围的小块区域
        cropped_mask = mask[loc]
        
        # 2. 在小块区域上创建二值掩膜
        binary_cropped_mask = (cropped_mask == organ_id).astype(np.float32)
        
        # 3. 对这个小块进行高质量缩放，得到概率图
        resized_prob_map = zoom(binary_cropped_mask, zoom_factors, order=3, mode='nearest')
        
        # 4. 计算这个缩放后的小块应该被放回大图的哪个位置
        start_coords = np.array([s.start for s in loc])
        new_start_coords = np.round(start_coords * zoom_factors).astype(int)
        
        # 5. 将缩放后的小块“贴”到4D volume的对应ID层
        end_coords = new_start_coords + np.array(resized_prob_map.shape)
        
        # 确保粘贴区域不越界
        paste_start = np.maximum(new_start_coords, 0)
        paste_end = np.minimum(end_coords, target_shape)
        
        # 计算需要从 resized_prob_map 中裁剪的区域
        crop_start = paste_start - new_start_coords
        crop_end = crop_start + (paste_end - paste_start)

        # 执行粘贴操作
        prob_volume[organ_id, 
                    paste_start[0]:paste_end[0], 
                    paste_start[1]:paste_end[1], 
                    paste_start[2]:paste_end[2]] = \
            resized_prob_map[crop_start[0]:crop_end[0], 
                             crop_start[1]:crop_end[1], 
                             crop_start[2]:crop_end[2]]

    # --- 重组步骤 ---
    # 通过 argmax 找到每个像素最终归属的ID
    final_mask = np.argmax(prob_volume, axis=0)
    
    # 如果最大概率也小于阈值，则视为背景
    max_probs = np.max(prob_volume, axis=0)
    final_mask[max_probs < 0.5] = 0
    
    return final_mask.astype(mask.dtype)'''
    
def resize_mask(mask, transform_pipeline):
    print("信息：正在使用 MONAI 管线执行高级语义保持 resize_mask()。")

    # 1. 准备工作
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

    # 2. 对每一个器官，在完整的、未裁剪的掩膜上进行操作
    for organ_id in organ_ids:
        # a. 创建一个与原始掩膜尺寸完全相同的全尺寸二值掩膜
        binary_mask_full_size = (mask == organ_id).astype(np.float32)

        # b. 【核心】将这个全尺寸的二值掩膜传递给 MONAI 进行缩放
        #    b1. 转换为 MONAI 期望的格式 (C, D, H, W)
        binary_mask_with_channel = np.expand_dims(binary_mask_full_size, axis=0)
        binary_mask_monai_format = np.transpose(binary_mask_with_channel, (0, 3, 2, 1))
        binary_tensor = torch.from_numpy(binary_mask_monai_format.copy())

        #    b2. 应用与影像完全相同的变换管线
        resized_tensor = transform_pipeline(binary_tensor)
        
        #    b3. 将结果转回我们熟悉的 Numpy 格式 (W, H, D)
        resized_numpy = resized_tensor.numpy()
        resized_prob_map = np.squeeze(np.transpose(resized_numpy, (0, 3, 2, 1)), axis=0)
        
        # c. 将得到的、尺寸正确的概率图直接放入概率volume的对应层
        prob_volume[organ_id] = resized_prob_map

    # 3. 最终决策
    final_mask = np.argmax(prob_volume, axis=0)
    max_probs = np.max(prob_volume, axis=0)
    final_mask[max_probs < 0.5] = 0
    
    return final_mask.astype(mask.dtype)


def parse_arguments():
    parser = argparse.ArgumentParser(description="处理并可视化3D医疗影像。")
    parser.add_argument('--visualize', action='store_true', 
                        help="如果设置此项，将在屏幕上显示可视化结果。")

    return parser.parse_args()


def main():
    args = parse_arguments()

    # 这里未来打算做一个扫描 + 遍历，subid只有1和2。
    #目前只做了单图片的选择。
    test_img_id = '100'
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