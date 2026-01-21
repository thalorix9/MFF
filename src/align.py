import cv2
import numpy as np

def align_images(images, ref_index=None, mode=cv2.MOTION_AFFINE):
    """
    使用 ECC 算法对齐多张图像。
    """
    if not images:
        return []
    
    n = len(images)
    if ref_index is None:
        ref_index = n // 2
    
    ref_img = images[ref_index]
    # 转换为灰度图用于计算变换矩阵
    if len(ref_img.shape) == 3:
        ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
    else:
        ref_gray = ref_img
        
    aligned_images = []
    
    # ECC 终止条件
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 1000, 1e-5)
    
    for i, img in enumerate(images):
        if i == ref_index:
            # 确保参考图像也是 BGRA 格式，保持通道一致性
            if img.shape[2] == 3:
                img_bgra = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
            else:
                img_bgra = img.copy()
            aligned_images.append(img_bgra)
            continue
            
        if len(img.shape) == 3:
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            img_gray = img
            
        try:
            # 初始化变换矩阵
            if mode == cv2.MOTION_HOMOGRAPHY:
                warp_matrix = np.eye(3, 3, dtype=np.float32)
            else:
                warp_matrix = np.eye(2, 3, dtype=np.float32)
            
            # 计算变换矩阵
            _, warp_matrix = cv2.findTransformECC(ref_gray, img_gray, warp_matrix, mode, criteria)
            
            # 准备 BGRA 图像进行变换
            if img.shape[2] == 3:
                img_bgra = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
            else:
                img_bgra = img.copy()

            # 应用变换，边界填充为全透明 (0,0,0,0)
            h, w = ref_img.shape[:2]
            if mode == cv2.MOTION_HOMOGRAPHY:
                aligned_img = cv2.warpPerspective(img_bgra, warp_matrix, (w, h), 
                                                flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                                                borderMode=cv2.BORDER_CONSTANT, 
                                                borderValue=(0, 0, 0, 0))
            else:
                aligned_img = cv2.warpAffine(img_bgra, warp_matrix, (w, h), 
                                           flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                                           borderMode=cv2.BORDER_CONSTANT, 
                                           borderValue=(0, 0, 0, 0))
                
            aligned_images.append(aligned_img)
            
        except cv2.error:
            # 配准失败时也转为 BGRA
            if img.shape[2] == 3:
                img_bgra = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
            else:
                img_bgra = img.copy()
            aligned_images.append(img_bgra)
            
    return aligned_images