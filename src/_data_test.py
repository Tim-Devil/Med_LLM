from data import load_data

def main():
    DATA_BASE_DIR = "/home/fpf/Med_LLM/data/CT_RATE/dataset"


    print("--- 测试1: 详细模式 (verbose=True) ---")
    data_dict_verbose = load_data(DATA_BASE_DIR, img_id='10', sub_id='1', verbose=True)
    if data_dict_verbose:
        print(f"函数返回成功。影像数据尺寸: {data_dict_verbose['image_data'].shape}")
    
    print("\n" + "="*60 + "\n")
    
    print("--- 测试2: 简洁模式 (verbose=False) ---")
    data_dict_simple = load_data(DATA_BASE_DIR, img_id='10', sub_id='2', verbose=False)
    if data_dict_simple:
        print(f"函数返回成功。掩膜数据尺寸: {data_dict_simple['mask_data'].shape}")

    print("\n" + "="*60 + "\n")

    print("--- 测试3: 文件不存在的情况 ---")
    load_data(DATA_BASE_DIR, img_id='999', sub_id='1', verbose=False)

if __name__ == "__main__":
    main()