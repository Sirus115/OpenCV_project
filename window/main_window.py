import os

from PyQt5.QtWidgets import QFileDialog, QSlider
from PyQt5 import uic
import cv2  # opencv读取的格式是BGR
import matplotlib.pyplot as plt
import numpy as np
from pyqt5_plugins.examplebuttonplugin import QtGui

import globalVar
from window import text_window


class Stats:

    def __init__(self):
        # 从文件中加载UI定义
        self.ui = uic.loadUi("ui/main.ui")
        # 加载 icon
        # app.setWindowIcon(QIcon('logo.png'))

        # 页面切换
        self.ui.btn_basic.clicked.connect(self.display1)
        self.ui.btn_smooth.clicked.connect(self.display2)
        self.ui.btn_operations.clicked.connect(self.display3)
        self.ui.btn_gradient.clicked.connect(self.display4)
        self.ui.btn_edge.clicked.connect(self.display5)
        self.ui.btn_histogram.clicked.connect(self.display6)
        self.ui.btn_spe.clicked.connect(self.display7)

        # slider初始化
        # 亮度调整
        self.ui.sld_bright.setMaximum(100)  # 设置最大最小值
        self.ui.sld_bright.setMinimum(-100)
        self.ui.sld_bright.setSingleStep(1)  # 设置单步值
        self.ui.sld_bright.setValue(0)  # 设置初始值
        self.ui.sld_bright.setTickPosition(QSlider.TicksBelow)  # 设置刻度线位置
        # 对比度调整
        self.ui.sld_contrast.setMaximum(25)  # 设置最大最小值
        self.ui.sld_contrast.setMinimum(-25)
        self.ui.sld_contrast.setSingleStep(1)  # 设置单步值
        self.ui.sld_contrast.setValue(0)  # 设置初始值
        self.ui.sld_contrast.setTickPosition(QSlider.TicksBelow)  # 设置刻度线位置

        # 绑定信号与槽
        # page1
        self.ui.btn_load.clicked.connect(self.btn_load_read)
        self.ui.cbx_color.currentIndexChanged.connect(self.cbx_color_change)
        self.ui.cbx_edge.currentIndexChanged.connect(self.cbx_edge_change)
        self.ui.btn_merge.clicked.connect(self.btn_merge_click)
        self.ui.btn_zoom.clicked.connect(self.btn_zoom_click)
        self.ui.btn_graph.clicked.connect(self.graph)
        self.ui.sld_bright.valueChanged.connect(self.brightness)
        self.ui.sld_contrast.valueChanged.connect(self.contrast)
        # page2
        self.ui.cbx_gray.currentIndexChanged.connect(self.cbx_gray_change)
        self.ui.cbx_smooth.currentIndexChanged.connect(self.smooth)
        # page3
        self.ui.btn_erode.clicked.connect(self.erode)
        self.ui.btn_dilate.clicked.connect(self.dilate)
        self.ui.cbx_cal.currentIndexChanged.connect(self.calculate)
        self.ui.btn_mor.clicked.connect(self.gradient)
        self.ui.cbx_hat.currentIndexChanged.connect(self.tophat)
        # page4
        self.ui.btn_sobel.clicked.connect(self.sobel)
        self.ui.btn_scharr.clicked.connect(self.scharr)
        self.ui.btn_lpc.clicked.connect(self.laplacian)
        self.ui.btn_canny.clicked.connect(self.canny)
        # page5
        self.ui.btn_gaussup.clicked.connect(self.Gaussup)
        self.ui.btn_gaussdown.clicked.connect(self.Gaussdown)
        self.ui.btn_lap.clicked.connect(self.lap)
        self.ui.btn_getedge.clicked.connect(self.getedge)
        self.ui.btn_mode.clicked.connect(self.mode)
        # page6
        self.ui.cbx_his.currentIndexChanged.connect(self.histogram)
        self.ui.cbx_fourier.currentIndexChanged.connect(self.fourier)
        # page7
        self.ui.btn_harris.clicked.connect(self.harris)
        self.ui.btn_sift.clicked.connect(self.sift)

        # 路径初始化
        self.curPath = os.path.abspath(os.path.dirname(__file__))
        self.rootPath = self.curPath[:self.curPath.find("OpenCV_project\\") + len("OpenCV_project\\")]

        self.result = self.rootPath + '/data/result/'
        globalVar.set_value('result', self.result)

    # 页面切换
    def display1(self):
        self.ui.stackedWidget.setCurrentIndex(0)

    def display2(self):
        self.ui.stackedWidget.setCurrentIndex(1)

    def display3(self):
        self.ui.stackedWidget.setCurrentIndex(2)

    def display4(self):
        self.ui.stackedWidget.setCurrentIndex(3)

    def display5(self):
        self.ui.stackedWidget.setCurrentIndex(4)

    def display6(self):
        self.ui.stackedWidget.setCurrentIndex(5)

    def display7(self):
        self.ui.stackedWidget.setCurrentIndex(6)

    '''------------------------------------------------图像基本操作-----------------------------------------------------'''

    # 读取数据
    def btn_load_read(self):
        self.ui.label_show.setText('结果显示')
        f = QFileDialog.getOpenFileName(None, "请选择读取的文件",
                                        self.rootPath, "JPG(*.jpg);;PNG(*.png);;MP4(*.mp4)")

        self.file = "".join(list(f[0]))
        globalVar.set_value('file', self.file)
        if self.file[-3:] != 'mp4':
            self.imshow(self.file, self.ui.img_origin)

        else:
            # 视频显示
            vc = cv2.VideoCapture('test.mp4')

            # 检查是否打开正确
            if vc.isOpened():
                open, frame = vc.read()
            else:
                open = False

            while open:
                ret, frame = vc.read()
                if frame is None:
                    break
                if ret == True:
                    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    cv2.imshow('result', frame)
                    if cv2.waitKey(100) & 0xFF == 27:
                        break  # 100ms播放视频慢，27代表按退出键退出
            vc.release()
            cv2.destroyAllWindows()

    # 颜色通道提取
    def cbx_color_change(self):
        self.ui.label_show.setText('结果显示')
        img = cv2.imread(self.file)
        b, g, r = cv2.split(img)
        method = self.ui.cbx_color.currentText()
        if method == 'Blue':
            self.ui.label_show.setText('Blue')
            cv2.imwrite(self.result + 'chapter1/' + 'Blue.jpg', b)
            self.imshow(self.result + 'chapter1/' + 'Blue.jpg', self.ui.img_new)
        elif method == 'Red':
            self.ui.label_show.setText('Red')
            cv2.imwrite(self.result + 'chapter1/' + 'Red.jpg', r)
            self.imshow(self.result + 'chapter1/' + 'Red.jpg', self.ui.img_new)
        elif method == 'Green':
            self.ui.label_show.setText('Green')
            cv2.imwrite(self.result + 'chapter1/' + 'Green.jpg', g)
            self.imshow(self.result + 'chapter1/' + 'Green.jpg', self.ui.img_new)

    # 边界填充
    def cbx_edge_change(self):
        self.ui.label_show.setText('结果显示')
        method = self.ui.cbx_edge.currentText()
        img = cv2.imread(self.file)
        top_size, bottom_size, left_size, right_size = (50, 50, 50, 50)
        replicate = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size,
                                       borderType=cv2.BORDER_REPLICATE)
        reflect = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, cv2.BORDER_REFLECT)
        reflect101 = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, cv2.BORDER_REFLECT_101)
        wrap = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, cv2.BORDER_WRAP)
        constant = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, cv2.BORDER_CONSTANT, value=0)

        if method == '复制法':
            self.ui.label_show.setText('复制法')
            cv2.imwrite(self.result + 'chapter1/' + 'replicate.jpg', replicate)
            self.imshow(self.result + 'chapter1/' + 'replicate.jpg', self.ui.img_new)
        elif method == '反射法':
            self.ui.label_show.setText('反射法')
            cv2.imwrite(self.result + 'chapter1/' + 'reflect.jpg', reflect)
            self.imshow(self.result + 'chapter1/' + 'reflect.jpg', self.ui.img_new)
        elif method == '反射法101':
            self.ui.label_show.setText('反射法101')
            cv2.imwrite(self.result + 'chapter1/' + 'reflect101.jpg', reflect101)
            self.imshow(self.result + 'chapter1/' + 'reflect101.jpg', self.ui.img_new)
        elif method == '外包装法':
            self.ui.label_show.setText('外包装法')
            cv2.imwrite(self.result + 'chapter1/' + 'wrap.jpg', wrap)
            self.imshow(self.result + 'chapter1/' + 'wrap.jpg', self.ui.img_new)
        elif method == '常量法':
            self.ui.label_show.setText('常量法')
            cv2.imwrite(self.result + 'chapter1/' + 'constant.jpg', constant)
            self.imshow(self.result + 'chapter1/' + 'constant.jpg', self.ui.img_new)

    # 图像融合
    def btn_merge_click(self):
        self.ui.label_show.setText('结果显示')
        f = QFileDialog.getOpenFileName(None, "请选择融合图片读来自的文件",
                                        self.rootPath, "JPG(*.jpg);;PNG(*.png);;MP4(*.mp4)")

        sec_route = "".join(list(f[0]))
        img_1 = cv2.imread(self.file)
        img_2 = cv2.imread(sec_route)
        img_1 = cv2.resize(img_1, (940, 940))
        img_2 = cv2.resize(img_2, (940, 940))

        res = cv2.addWeighted(img_1, 0.4, img_2, 0.6, 0)  # 将两幅图像按照不同的比例合成一张图像
        cv2.imwrite(self.result + 'chapter1/' + 'merge.jpg', res)
        self.imshow(self.result + 'chapter1/' + 'merge.jpg', self.ui.img_new)

    # 图片缩放
    def btn_zoom_click(self):
        self.ui.label_show.setText('结果显示')
        img = cv2.imread(self.file)
        fx = float(self.ui.txt_fx.text())
        fy = float(self.ui.txt_fy.text())
        # if fx == 0 or fy == 0:
        #     self.ui.btn_zoom.setEnabled(False)
        res = cv2.resize(img, (0, 0), fx=fx, fy=fy)
        cv2.imwrite(self.result + 'chapter1/' + 'zoom.jpg', res)
        self.imshow(self.result + 'chapter1/' + 'zoom.jpg', self.ui.img_new)

    # 亮度调节
    def brightness(self):
        self.ui.label_show.setText('结果显示')
        img = cv2.imread(self.file)
        level = self.ui.sld_bright.value()
        self.ui.lbl_bri.setText(str(level))
        # 创建一个与原图像一样大小的空白图像
        blank = np.zeros_like(img)
        if level >= 0:
            # 将原图像和空白图像相加即可增加亮度
            blank[:, :] = (level, level, level)  # 空白图像的bgr都为50，这里增加或者减小值
            result = cv2.add(img, blank)
        else:
            # 将原图像和空白图像相减即可减小亮度
            blank[:, :] = (-level, -level, -level)
            result = cv2.subtract(img, blank)
        cv2.imwrite(self.result + 'chapter1/' + 'brightness.jpg', result)
        self.imshow(self.result + 'chapter1/' + 'brightness.jpg', self.ui.img_new)

    # 对比度调节
    def contrast(self):
        self.ui.label_show.setText('结果显示')
        img = cv2.imread(self.file)
        level = self.ui.sld_contrast.value()
        self.ui.lbl_con.setText(str(level))
        # 创建一个与原图像一样大小的空白图像
        blank = np.zeros_like(img)
        if level >= 0:
            blank[:, :] = (level, level, level)
            # 将原图像和空白图像相乘即可增加对比度
            result = cv2.multiply(img, blank)
        else:
            blank[:, :] = (-level, -level, -level)
            # 将原图像和空白图像相除即可减小对比度
            result = cv2.divide(img, blank)
        cv2.imwrite(self.result + 'chapter1/' + 'contrast.jpg', result)
        self.imshow(self.result + 'chapter1/' + 'contrast.jpg', self.ui.img_new)

    # 直方图均衡化
    def graph(self):
        self.ui.label_show.setText('结果显示')
        img = cv2.imread(self.file)
        # 彩色图像均衡化
        (b, g, r) = cv2.split(img)
        bH = cv2.equalizeHist(b)
        gH = cv2.equalizeHist(g)
        rH = cv2.equalizeHist(r)

        # 合并每一个通道
        result = cv2.merge((bH, gH, rH))
        cv2.imwrite(self.result + 'chapter1/' + 'Histogram Equalization.jpg', result)
        self.imshow(self.result + 'chapter1/' + 'Histogram Equalization.jpg', self.ui.img_new)

    '''------------------------------------------阈值与平滑处理------------------------------------------------------'''

    def cbx_gray_change(self):
        self.ui.label_show.setText('结果显示')
        method = self.ui.cbx_gray.currentText()
        img_gray = cv2.imread(self.file)
        if method == 'cv2.THRESH_BINARY':
            # 输入图，只能输入单通道图像，通常来说为灰度图
            self.ui.label_show.setText('cv2.THRESH_BINARY')
            ret, result = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
        elif method == 'cv2.THRESH_BINARY_INV':
            self.ui.label_show.setText('cv2.THRESH_BINARY_INV')
            ret, result = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY_INV)
        elif method == 'cv2.THRESH_TRUNC':
            self.ui.label_show.setText('cv2.THRESH_TRUNC')
            ret, result = cv2.threshold(img_gray, 127, 255, cv2.THRESH_TRUNC)
        elif method == 'cv2.THRESH_TOZERO':
            self.ui.label_show.setText('cv2.THRESH_TOZERO')
            ret, result = cv2.threshold(img_gray, 127, 255, cv2.THRESH_TOZERO)
        elif method == 'cv2.THRESH_TOZERO_INV':
            self.ui.label_show.setText('cv2.THRESH_TOZERO_INV')
            ret, result = cv2.threshold(img_gray, 127, 255, cv2.THRESH_TOZERO_INV)

        cv2.imwrite(self.result + 'chapter2/' + 'img_gray.jpg', result)
        self.imshow(self.result + 'chapter2/' + 'img_gray.jpg', self.ui.img_new)

    def smooth(self):
        self.ui.label_show.setText('结果显示')
        img = cv2.imread(self.file)
        method = self.ui.cbx_smooth.currentText()
        if method == '均值滤波':
            self.ui.label_show.setText('均值滤波结果')
            result = cv2.blur(img, (3, 3))
        elif method == '方框滤波':
            # 容易越界
            self.ui.label_show.setText('方框滤波结果')
            result = cv2.boxFilter(img, -1, (3, 3), normalize=False)
        elif method == '高斯滤波':
            self.ui.label_show.setText('高斯滤波结果')
            result = cv2.GaussianBlur(img, (5, 5), 1)
        elif method == '中值滤波':
            self.ui.label_show.setText('中值滤波结果')
            result = cv2.medianBlur(img, 5)
        cv2.imwrite(self.result + 'chapter2/' + 'img_smooth.jpg', result)
        self.imshow(self.result + 'chapter2/' + 'img_smooth.jpg', self.ui.img_new)

    '''------------------------------------------图像形态学处理(腐蚀、膨胀)-----------------------------------------------'''

    # 腐蚀
    def erode(self):
        self.ui.label_show.setText('结果显示')
        # 第一个效果更好
        img = cv2.imread(self.file)
        kernel = np.ones((3, 3), np.uint8)
        erosion = cv2.erode(img, kernel, iterations=1)
        # 第二个可以只保存，不展示
        kernel = np.ones((30, 30), np.uint8)
        erosion_1 = cv2.erode(img, kernel, iterations=1)
        erosion_2 = cv2.erode(img, kernel, iterations=2)
        erosion_3 = cv2.erode(img, kernel, iterations=3)
        res = np.hstack((erosion_1, erosion_2, erosion_3))
        cv2.imwrite(self.result + 'chapter3/' + 'erosion.jpg', erosion)
        cv2.imwrite(self.result + 'chapter3/' + '3_erosion.jpg', res)
        self.imshow(self.result + 'chapter3/' + 'erosion.jpg', self.ui.img_new)

    # 膨胀
    def dilate(self):
        self.ui.label_show.setText('结果显示')
        img = cv2.imread(self.file)
        kernel = np.ones((30, 30), np.uint8)
        dilate_1 = cv2.dilate(img, kernel, iterations=1)
        dilate_2 = cv2.dilate(img, kernel, iterations=2)
        dilate_3 = cv2.dilate(img, kernel, iterations=3)
        res = np.hstack((dilate_1, dilate_2, dilate_3))
        cv2.imwrite(self.result + 'chapter3/' + '3_dilate.jpg', res)
        cv2.imwrite(self.result + 'chapter3/' + 'dilate.jpg', dilate_1)
        self.imshow(self.result + 'chapter3/' + 'dilate.jpg', self.ui.img_new)

    # 开运算与闭运算
    def calculate(self):
        img = cv2.imread(self.file)
        method = self.ui.cbx_cal.currentText()
        if method == '开运算':
            self.ui.label_show.setText('先腐蚀，再膨胀（去毛刺，再回复粗度）')
            kernel = np.ones((5, 5), np.uint8)
            result = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        elif method == '闭运算':
            self.ui.label_show.setText('先膨胀，再腐蚀（不能去毛刺）')
            kernel = np.ones((5, 5), np.uint8)
            result = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        cv2.imwrite(self.result + 'chapter3/' + 'after_calculate.jpg', result)
        self.imshow(self.result + 'chapter3/' + 'after_calculate.jpg', self.ui.img_new)

    # 梯度运算
    def gradient(self):
        self.ui.label_show.setText('结果显示')
        img = cv2.imread(self.file)
        kernel = np.ones((5, 5), np.uint8)
        gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
        cv2.imwrite(self.result + 'chapter3/' + 'gradient.jpg', gradient)
        self.imshow(self.result + 'chapter3/' + 'gradient.jpg', self.ui.img_new)

    # 礼帽与黑帽
    def tophat(self):
        self.ui.label_show.setText('结果显示')
        img = cv2.imread(self.file)
        method = self.ui.cbx_hat.currentText()
        kernel = np.ones((5, 5), np.uint8)
        if method == '礼帽':
            self.ui.label_show.setText('礼帽 = 原始输入 - 开运算结果')
            hat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
        elif method == '黑帽':
            self.ui.label_show.setText('黑帽 = 闭运算 - 原始输入')
            hat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)
        cv2.imwrite(self.result + 'chapter3/' + 'img_hat.jpg', hat)
        self.imshow(self.result + 'chapter3/' + 'img_hat.jpg', self.ui.img_new)

    '''----------------------------------------------图像梯度处理-------------------------------------------------------'''

    def sobel(self):
        self.ui.label_show.setText('结果显示')
        img = cv2.imread(self.file)
        # 分别计算x、y方向梯度，再合并
        sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        sobelx = cv2.convertScaleAbs(sobelx)
        sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        sobely = cv2.convertScaleAbs(sobely)
        sobelxy = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)
        cv2.imwrite(self.result + 'chapter4/' + 'sobel.jpg', sobelxy)
        self.imshow(self.result + 'chapter4/' + 'sobel.jpg', self.ui.img_new)

    def scharr(self):
        self.ui.label_show.setText('结果显示')
        img = cv2.imread(self.file)
        scharrx = cv2.Scharr(img, cv2.CV_64F, 1, 0)
        scharry = cv2.Scharr(img, cv2.CV_64F, 0, 1)
        scharrx = cv2.convertScaleAbs(scharrx)
        scharry = cv2.convertScaleAbs(scharry)
        scharrxy = cv2.addWeighted(scharrx, 0.5, scharry, 0.5, 0)
        cv2.imwrite(self.result + 'chapter4/' + 'scharr.jpg', scharrxy)
        self.imshow(self.result + 'chapter4/' + 'scharr.jpg', self.ui.img_new)

    def laplacian(self):
        self.ui.label_show.setText('结果显示')
        img = cv2.imread(self.file)
        laplacian = cv2.Laplacian(img, cv2.CV_64F)
        laplacian = cv2.convertScaleAbs(laplacian)
        cv2.imwrite(self.result + 'chapter4/' + 'laplacian.jpg', laplacian)
        self.imshow(self.result + 'chapter4/' + 'laplacian.jpg', self.ui.img_new)

    def canny(self):
        self.ui.label_show.setText('结果显示')
        img = cv2.imread(self.file)
        result = cv2.Canny(img, 50, 100)
        cv2.imwrite(self.result + 'chapter4/' + 'canny.jpg', result)
        self.imshow(self.result + 'chapter4/' + 'canny.jpg', self.ui.img_new)

    '''----------------------------------------图像金字塔与轮廓检测(模板匹配)----------------------------------------------'''

    # 高斯 上采样
    def Gaussup(self):
        img = cv2.imread(self.file)
        up = cv2.pyrUp(img)
        self.ui.label_origin.setText('原始图像' + str(img.shape))
        self.ui.label_show.setText('处理结果' + str(up.shape))

        up_down = cv2.pyrDown(cv2.pyrUp(img))
        cv2.imwrite(self.result + 'chapter5/' + 'Gauss_up_down.jpg', up_down)
        self.imshow(self.result + 'chapter5/' + 'Gauss_up_down.jpg', self.ui.img_new)

    # 高斯 下采样
    def Gaussdown(self):
        img = cv2.imread(self.file)
        down = cv2.pyrDown(img)
        self.ui.label_origin.setText('原始图像' + str(img.shape))
        self.ui.label_show.setText('处理结果' + str(down.shape))

        down_up = cv2.pyrUp(cv2.pyrDown(img))
        cv2.imwrite(self.result + 'chapter5/' + 'Gauss_down_up.jpg', down_up)
        self.imshow(self.result + 'chapter5/' + 'Gauss_down_up.jpg', self.ui.img_new)

    # 拉普拉斯金字塔
    def lap(self):
        self.ui.label_show.setText('结果显示')
        img = cv2.imread(self.file)
        down = cv2.pyrDown(img)
        down_up = cv2.pyrUp(down)
        l_1 = img - down_up
        cv2.imwrite(self.result + 'chapter5/' + 'Laplacian_pyramid.jpg', l_1)
        self.imshow(self.result + 'chapter5/' + 'Laplacian_pyramid.jpg', self.ui.img_new)

    # 图像轮廓
    def getedge(self):
        self.ui.label_show.setText('结果显示')
        img = cv2.imread(self.file)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        # 计算出嵌套轮廓
        # 这里新版本的cv2.findContours返回的只有两个值
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        # 传入绘制图像，轮廓，轮廓索引(-1全部)，颜色模式，线条厚度
        draw_img = img.copy()  # 注意需要copy,要不原图会变
        res = cv2.drawContours(draw_img, contours, -1, (0, 0, 255), 2)

        cv2.imwrite(self.result + 'chapter5/' + 'edge.jpg', res)
        self.imshow(self.result + 'chapter5/' + 'edge.jpg', self.ui.img_new)

    # 模板匹配
    def mode(self):
        self.ui.label_show.setText('识别结果')
        img = cv2.imread(self.file)
        f = QFileDialog.getOpenFileName(None, "请选择融合图片读来自的文件",
                                        self.rootPath, "JPG(*.jpg);;PNG(*.png);;MP4(*.mp4)")

        sec_route = "".join(list(f[0]))
        template = cv2.imread(sec_route, 0)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = template.shape[:2]
        res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
        threshold = 0.8
        # 取匹配程度大于%80的坐标
        loc = np.where(res >= threshold)
        for pt in zip(*loc[::-1]):  # *号表示可选参数
            bottom_right = (pt[0] + w, pt[1] + h)
            cv2.rectangle(img, pt, bottom_right, (0, 255, 0), 1)
        cv2.imwrite(self.result + 'chapter5/' + 'template matching.jpg', img)
        self.imshow(self.result + 'chapter5/' + 'template matching.jpg', self.ui.img_new)

    '''------------------------------------------直方图与傅里叶变换------------------------------------------------------'''

    # 图像直方图
    def histogram(self):
        # img = cv2.imread(self.file)
        method = self.ui.cbx_his.currentText()
        if method == '直方图':
            img = cv2.imread(self.file, 0)
            hist = cv2.calcHist([img], [0], None, [256], [0, 256])  # (256, 1)
            img = plt.hist(img.ravel(), 256)
            plt.savefig(self.result + 'chapter6/' + 'histogram.jpg')
            self.imshow(self.result + 'chapter6/' + 'histogram.jpg', self.ui.img_new)
        elif method == '三通道直方图':
            img = cv2.imread(self.file)
            color = ('blue', 'green', 'red')
            for i, color in enumerate(color):
                hist = cv2.calcHist([img], [i], None, [256], [0, 256])
                plt.plot(hist, color=color)
                plt.xlim([0, 256])
            plt.savefig(self.result + 'chapter6/' + 'Three channel histogram.jpg')
            self.imshow(self.result + 'chapter6/' + 'Three channel histogram.jpg', self.ui.img_new)
        elif method == '自适应直方图均衡化':
            # 做均衡化
            img = cv2.imread(self.file, 0)
            equ = cv2.equalizeHist(img)
            plt.hist(equ.ravel(), 256)

            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            res_clahe = clahe.apply(
                img)  # apply(func,*args,**kwargs)的返回值就是func()的返回值，apply()的元素参数是有序的，元素的顺序必须和func()形式参数的顺序一致
            res = np.hstack((img, equ, res_clahe))
            cv2.imwrite(self.result + 'chapter5/' + 'Adaptive histogram equalization.jpg', res)
            self.imshow(self.result + 'chapter5/' + 'Adaptive histogram equalization.jpg', self.ui.img_new)

    # 傅里叶变换
    def fourier(self):
        img = cv2.imread(self.file, 0)
        method = self.ui.cbx_fourier.currentText()
        img_float32 = np.float32(img)
        if method == '低通滤波':
            # 低频
            dft = cv2.dft(img_float32, flags=cv2.DFT_COMPLEX_OUTPUT)
            dft_shift = np.fft.fftshift(dft)

            rows, cols = img.shape
            crow, ccol = int(rows / 2), int(cols / 2)  # 中心位置

            # 低通滤波
            mask = np.zeros((rows, cols, 2), np.uint8)
            mask[crow - 30:crow + 30, ccol - 30:ccol + 30] = 1

            # IDFT
            fshift = dft_shift * mask
            f_ishift = np.fft.ifftshift(fshift)
            img_back = cv2.idft(f_ishift)
            img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

            plt.imshow(img_back, cmap='gray')
            plt.savefig(self.result + 'chapter6/' + 'Low pass filtering.jpg')
            self.imshow(self.result + 'chapter6/' + 'Low pass filtering.jpg', self.ui.img_new)
        elif method == '高通滤波':
            # 高频
            dft = cv2.dft(img_float32, flags=cv2.DFT_COMPLEX_OUTPUT)
            dft_shift = np.fft.fftshift(dft)

            rows, cols = img.shape
            crow, ccol = int(rows / 2), int(cols / 2)  # 中心位置

            # 高通滤波
            mask = np.ones((rows, cols, 2), np.uint8)
            mask[crow - 30:crow + 30, ccol - 30:ccol + 30] = 0

            # IDFT
            fshift = dft_shift * mask
            f_ishift = np.fft.ifftshift(fshift)
            img_back = cv2.idft(f_ishift)
            img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

            plt.imshow(img_back, cmap='gray')
            plt.savefig(self.result + 'chapter6/' + 'High pass filtering.jpg')
            self.imshow(self.result + 'chapter6/' + 'High pass filtering.jpg', self.ui.img_new)

    '''-----------------------------------------------图像特征---------------------------------------------------------'''

    # harris 特征
    def harris(self):
        img = cv2.imread(self.file)
        self.ui.label_origin.setText('原始尺寸' + str(img.shape))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # gray = np.float32(gray)
        dst = cv2.cornerHarris(gray, 10, 5, 0.04)
        self.ui.label_show.setText('结果尺寸' + str(dst.shape))

        img[dst > 0.01 * dst.max()] = [0, 0, 255]
        cv2.imwrite(self.result + 'chapter7/' + 'harris.jpg', img)
        self.imshow(self.result + 'chapter7/' + 'harris.jpg', self.ui.img_new)

    # SIFT
    def sift(self):
        img = cv2.imread(self.file)
        key_points = img.copy()
        # 实例化SIFT算法
        sift = cv2.SIFT_create()
        # 得到特征点
        kp = sift.detect(img, None)
        # 绘制特征点
        cv2.drawKeypoints(img, kp, key_points)
        # 计算特征
        kp, des = sift.compute(img, kp)
        self.ui.label_show.setText('特征点绘制')
        file = open(file=self.result + 'chapter7/' + 'keypoint.txt', mode='w', encoding='utf-8')
        file.write(str(des[0]))
        file.close()
        cv2.imwrite(self.result + 'chapter7/' + 'SIFT_keypoint.jpg', key_points)
        self.imshow(self.result + 'chapter7/' + 'SIFT_keypoint.jpg', self.ui.img_new)

        self.window = text_window.Stats()
        self.window.ui.show()

    # 定义图片显示函数
    def imshow(self, file, label):
        jpg = QtGui.QPixmap(file).scaled(label.width(), label.height())
        label.setPixmap(jpg)
