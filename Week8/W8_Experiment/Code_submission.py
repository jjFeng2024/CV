# Step1 图像加载与预处理
def load_image(img1_path, img2_path):
    img1 = cv.imread(img1_path)
    img2 = cv.imread(img2_path)
    # 统一分辨率
    rate = 600 / img1.shape[1]
    img1 = cv.resize(img1, (int(rate*img1.shape[1]), int(rate*img1.shape[0])))
    img2 = cv.resize(img2, (img1.shape[1], img1.shape[0]))
    # 边界填充（用于拼接时留出空间）
    img1 = cv.copyMakeBorder(img1, 250,250,250,250, cv.BORDER_CONSTANT)
    img2 = cv.copyMakeBorder(img2, 250,250,250,250, cv.BORDER_CONSTANT)
    return img1, img2

# Step2  特征提取与匹配
def match_feature_point(img1, img2):
    # SIFT特征检测
    sift = cv.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # FLANN匹配器
    flann = cv.FlannBasedMatcher(
        dict(algorithm=1, trees=5),  # KD-Tree索引
        dict(checks=50)  # 搜索次数
    )
    matches = flann.knnMatch(des1, des2, k=2)  # KNN匹配
    return kp1, kp2, matches

# Step3 匹配点提纯与单应性矩阵计算
def get_good_match(img1, img2, kp1, kp2, matches):
    # 比率测试筛选匹配点
    good_match = []
    for m, n in matches:
        if m.distance < 0.5 * n.distance:  # Lowe's ratio test
            good_match.append(m)

    # 计算单应性矩阵（RANSAC提纯）
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_match]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_match]).reshape(-1, 1, 2)
    M, _ = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)

    # 图像配准
    img2_warped = cv.warpPerspective(img2, M, (img2.shape[1], img2.shape[0]))

    # 简单拼接（直接覆盖）
    dst = img1.copy()
    dst[img2_warped > 0] = img2_warped[img2_warped > 0]

    return M, dst

# Step4 图像融合（加权混合）
def blend_image(img1, img2):
    rows, cols = img1.shape[:2]
    result = np.zeros((rows, cols, 3), np.uint8)

    # 找到重叠区域边界
    left = next(col for col in range(cols) if img1[:, col].any() and img2[:, col].any())
    right = next(col for col in reversed(range(cols)) if img1[:, col].any() and img2[:, col].any())

    # 线性加权融合
    for col in range(cols):
        if img1[:, col].any() and img2[:, col].any():
            alpha = (col - left) / (right - left)  # 动态权重
            result[:, col] = img1[:, col] * alpha + img2[:, col] * (1 - alpha)
        elif img1[:, col].any():
            result[:, col] = img1[:, col]
        else:
            result[:, col] = img2[:, col]
    return result
