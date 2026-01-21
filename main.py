import os
import cv2
import glob
import logging
from src.align import align_images
from src.fuse import simple_focus_fusion

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def main():
    setup_logging()
    
    # --- 配置路径 (参考 JointRF-master 结构) ---
    # 输入目录：存放多个子文件夹，每个子文件夹是一组待融合图片
    base_input_dir = "UnregisteredImages"  
    # 输出目录：存放处理结果
    base_output_dir = "Results"            
    
    if not os.path.exists(base_output_dir):
        os.makedirs(base_output_dir)
    
    if not os.path.exists(base_input_dir):
        os.makedirs(base_input_dir)
        logging.warning(f"Created '{base_input_dir}'. Please copy your image folders (e.g., 'G1', '003_008') into it.")
        return

    # 获取所有子目录（每一组数据）
    # 过滤掉非文件夹项
    groups = [d for d in os.listdir(base_input_dir) if os.path.isdir(os.path.join(base_input_dir, d))]
    
    if not groups:
        logging.warning(f"No subdirectories found in '{base_input_dir}'. Please organize images into subfolders (e.g., UnregisteredImages/Group1/).")
        return

    logging.info(f"Found {len(groups)} groups to process: {groups}")

    # --- 批量处理 ---
    for group_name in groups:
        group_path = os.path.join(base_input_dir, group_name)
        logging.info(f"Processing group: {group_name}")
        
        # 读取该组下的所有图片
        image_paths = sorted(glob.glob(os.path.join(group_path, "*.[jp][pn]g")))
        
        if len(image_paths) < 2:
            logging.warning(f"  Skipping {group_name}: Need at least 2 images, found {len(image_paths)}.")
            continue
            
        # 加载图片
        images = []
        for p in image_paths:
            img = cv2.imread(p)
            if img is not None:
                images.append(img)
            else:
                logging.error(f"  Could not read {p}")
        
        if not images:
            continue

        # 1. 对齐 (Alignment)
        logging.info("  Step 1: Aligning images...")
        try:
            # 使用仿射变换模式 (MOTION_AFFINE) 适应微抖动（平移+旋转+微缩放）
            aligned_images = align_images(images, mode=cv2.MOTION_AFFINE)
        except Exception as e:
            logging.error(f"  Alignment failed for {group_name}: {e}")
            continue
        
        # 2. 融合 (Fusion)
        logging.info("  Step 2: Fusing images...")
        try:
            fused_image = simple_focus_fusion(aligned_images)
        except Exception as e:
            logging.error(f"  Fusion failed for {group_name}: {e}")
            continue
        
        if fused_image is not None:
            # 3. 保存结果
            # 在 Results 下创建同名子文件夹，方便管理
            group_output_dir = os.path.join(base_output_dir, group_name)
            if not os.path.exists(group_output_dir):
                os.makedirs(group_output_dir)
                
            # 保存融合结果
            save_path = os.path.join(group_output_dir, "fused.png")
            cv2.imwrite(save_path, fused_image)
            logging.info(f"  Success! Result saved to: {save_path}")
            
            # (可选) 保存对齐后的中间结果，用于调试或作为训练数据的 Ground Truth
            # for i, img in enumerate(aligned_images):
            #     cv2.imwrite(os.path.join(group_output_dir, f"aligned_{i}.png"), img)
        else:
            logging.error("  Fusion returned None.")

    logging.info("All tasks finished.")

if __name__ == "__main__":
    main()