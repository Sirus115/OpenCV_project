import numpy as np
import cv2

# 读取图片
img = cv2.imread("face.jpg")
key_points = img.copy()

# 实例化SIFT算法
sift = cv2.SIFT_create()

# 得到特征点
kp = sift.detect(img, None)
print(np.array(kp).shape)

# 绘制特征点
cv2.drawKeypoints(img, kp, key_points)

# 图片展示
cv2.imshow("key points", key_points)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 保存图片
cv2.imwrite("key_points.jpg", key_points)

# 计算特征
kp, des = sift.compute(img, kp)

# 调试输出
print(des.shape)
print(des[0])
