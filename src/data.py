import nibabel as nib
import numpy as np
import os

def load_data(base_dir, img_id, sub_id, verbose=False):

    folder_name = f"train_{img_id}"
    subfolder_name = f"train_{img_id}_a"
    file_name = f"train_{img_id}_a_{sub_id}.nii.gz"

    img_id = str(img_id)
    sub_id = str(sub_id)
    # 必要的鲁棒性处理

    image_path = os.path.join(base_dir, "train_fixed", folder_name, subfolder_name, file_name)
    mask_path = os.path.join(base_dir, "ts_seg", "ts_total", "train_fixed", folder_name, subfolder_name, file_name)

    print(f"正在加载: {file_name}...")

    try:
        if verbose: print(f"正在处理影像")

        image_nii = nib.load(image_path)
        image_data = image_nii.get_fdata()

        # 2. 加载掩膜
        if verbose: print(f"正在处理掩膜")
        mask_nii = nib.load(mask_path)
        mask_data = mask_nii.get_fdata()

        # 3. 验证尺寸是否匹配
        if image_data.shape != mask_data.shape:
            print(f"警告: 文件 {file_name} 的影像和掩膜尺寸不匹配！")
            
        # 4. 根据 verbose 参数决定是否打印详细信息
        if verbose:
            print("--- 影像信息 ---")
            print(f"影像尺寸 (Shape): {image_data.shape}")
            print(f"数据类型 (dtype): {image_data.dtype}")
            print(f"影像强度范围: Min={np.min(image_data):.2f}, Max={np.max(image_data):.2f}")
            print(f"头部信息 (Affine):\n{image_nii.affine}\n")
            
            print("--- 掩膜信息 ---")
            print(f"掩膜尺寸 (Shape): {mask_data.shape}")
            print(f"数据类型 (dtype): {mask_data.dtype}")
            unique_labels = np.unique(mask_data)
            print(f"掩膜中的唯一标签值: {unique_labels}")
            print("-" * 50)

        if not verbose:
            print(f"加载完成: {file_name}")

        # 5. 返回包含所有必要数据的字典
        return {
            "image_data": image_data,
            "mask_data": mask_data,
            "affine": image_nii.affine,
            "file_name": file_name
        }

    except FileNotFoundError:
        print(f"错误: 找不到文件 {image_path} 或 {mask_path}")
        return None
    except Exception as e:
        print(f"处理文件 {file_name} 时发生未知错误: {e}")
        return None