import json
import numpy as np
from scipy.ndimage import find_objects

import os

def extract_organ_bboxes(mask, img_id, output_path):
    # 获取唯一的器官ID，注意排除背景(默认为0)
    organ_ids = np.unique(mask)
    organ_ids = organ_ids[organ_ids != 0]
    
    # 创建JSON对象
    json_data = {img_id: {}}
    
    # 对每个器官ID
    for organ_id in organ_ids:
        # 创建一个二值掩码，只包含当前器官
        binary_mask = (mask == organ_id)
        
        # 找到非零区域的边界框
        regions = find_objects(binary_mask)
        
        # 如果找到区域
        if regions and len(regions) > 0:
            region = regions[0]
            # 获取区域切片
            z_slice, y_slice, x_slice = region
            
            # 转换为边界框坐标 [x1, y1, z1, x2, y2, z2]
            bbox = [
                int(x_slice.start), int(y_slice.start), int(z_slice.start),
                int(x_slice.stop), int(y_slice.stop), int(z_slice.stop)
            ]
            
            # 添加边界框到结果字典，使用器官ID作为键
            json_data[img_id][str(int(organ_id))] = bbox
    
    # 保存JSON文件
    with open(output_path, 'w') as f:
        json.dump(json_data, f, indent=4)
    
    print(f"器官边界框信息已保存至: {output_path}")