import cv2
import numpy as np

image = cv2.imread('E:\Code\OpenCV_project\data\kitty.jpg')
cv2.imshow("input", image)

# 创建一个与原图像一样大小的空白图像
blank = np.zeros_like(image)
blank[:, :] = (-20, -20, -20)  # 空白图像的bgr都为50，这里增加或者减小值

# 将原图像和空白图像相加即可增加亮度
result_1 = cv2.add(image,blank)
cv2.imshow("result_1",result_1)

# 将原图像和空白图像相减即可减小亮度
# result_2 = cv2.subtract(image, blank)
# cv2.imshow("result_1", result_2)
cv2.waitKey(0)
cv2.destroyAllWindows()