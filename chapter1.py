import cv2
import numpy as np

image = cv2.imread('E:\Code\OpenCV_project\data\kitty.jpg')
cv2.imshow("input", image)

# 创建一个与原图像一样大小的空白图像
blank = np.zeros_like(image)
blank[:, :] = (100, 100, 100)  # bgr 分别为2，即为图像对比度比例

# # 将原图像和空白图像相乘即可增加对比度
# result_1 = cv2.multiply(image,blank)
# cv2.imshow("result_1",result_1)

# 将原图像和空白图像相除即可减小对比度
# result_2 = cv2.divide(image, blank)
# cv2.imshow("result_2", result_2)

cv2.waitKey(0)
cv2.destroyAllWindows()
