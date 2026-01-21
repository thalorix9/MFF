import cv2
import numpy as np

def compute_laplacian_focus_measure(img):
    """
    计算拉普拉斯能量图（Laplacian Energy），值越大表示越清晰（边缘越锐利）。
    """
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return np.abs(lap)

def simple_focus_fusion(images):
    """
    基于拉普拉斯能量的加权融合或最大值选择融合。
    """
    if not images:
        return None
    
    # 1. 计算所有图的清晰度图
    score_maps = []
    
    for img in images:
        # 分离 alpha 通道
        if img.shape[2] == 4:
            bgr = img[:, :, :3]
            alpha = img[:, :, 3]
            
            # 严格阈值：只保留完全不透明的区域 (>250)，排除插值产生的半透明边缘
            # 并进行腐蚀操作 (erode)，进一步向内收缩 1-2 像素，彻底去除边缘的暗色插值伪影
            binary_mask = (alpha > 250).astype(np.uint8)
            kernel = np.ones((3, 3), np.uint8)
            # 腐蚀1次通常足够去除插值边缘
            mask = cv2.erode(binary_mask, kernel, iterations=1).astype(np.float64)
        else:
            bgr = img
            mask = np.ones(img.shape[:2], dtype=np.float64)
            
        s_map = compute_laplacian_focus_measure(bgr)
        # 高斯模糊平滑决策图
        s_map = cv2.GaussianBlur(s_map, (5, 5), 0)
        
        # 关键：如果该像素被Mask掉，强制其清晰度得分为 -1e9
        s_map = s_map * mask - (1 - mask) * 1e9
        
        score_maps.append(s_map)
        
    score_maps = np.array(score_maps) # (N, H, W)
    
    # 2. 逐像素选择最清晰的图
    best_indices = np.argmax(score_maps, axis=0) # (H, W)
    
    images_array = np.array(images) # (N, H, W, C)
    
    # 处理通道数
    c = images[0].shape[2]
    h, w = images[0].shape[:2]
    
    # 扩展索引维度
    indices_expanded = np.repeat(best_indices[:, :, np.newaxis], c, axis=2)
    m, n = np.ogrid[:h, :w]
    result = images_array[indices_expanded, m[:, :, np.newaxis], n[:, :, np.newaxis], np.arange(c)]
    
    return result