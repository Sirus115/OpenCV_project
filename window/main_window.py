import os

from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QPlainTextEdit, QMessageBox, QLabel, QFileDialog, \
    QSlider
from PyQt5 import uic
from PyQt5.QtGui import QIcon, QImage, QPixmap
import cv2  # opencv读取的格式是BGR
import matplotlib.pyplot as plt
import numpy as np
from PySide2.QtUiTools import QUiLoader
from pyqt5_plugins.examplebuttonplugin import QtGui

import globalVar


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

        # slider初始化
        # 亮度调整
        self.ui.sld_bright.setMaximum(100)  # 设置最大最小值
        self.ui.sld_bright.setMinimum(-100)
        self.ui.sld_bright.setSingleStep(1)  # 设置单步值
        self.ui.sld_bright.setValue(0)  # 设置初始值
        self.ui.sld_bright.setTickPosition(QSlider.TicksBelow)  # 设置刻度线位置
        # 对比度调整
        self.ui.sld_contrast.setMaximum(50)  # 设置最大最小值
        self.ui.sld_contrast.setMinimum(-50)
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

    '''------------------------------------------------图像基本操作-----------------------------------------------------'''

    # 读取数据
    def btn_load_read(self):
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
        img = cv2.imread(self.file)
        fx = float(self.ui.txt_fx.text())
        fy = float(self.ui.txt_fy.text())
        res = cv2.resize(img, (0, 0), fx=fx, fy=fy)
        cv2.imwrite(self.result + 'chapter1/' + 'zoom.jpg', res)
        self.imshow(self.result + 'chapter1/' + 'zoom.jpg', self.ui.img_new)

    # 亮度调节
    def brightness(self):
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
        img = cv2.imread(self.file)
        kernel = np.ones((5, 5), np.uint8)
        gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
        cv2.imwrite(self.result + 'chapter3/' + 'gradient.jpg', gradient)
        self.imshow(self.result + 'chapter3/' + 'gradient.jpg', self.ui.img_new)

    # 礼帽与黑帽
    def tophat(self):
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

    # 定义图片显示函数
    def imshow(self, file, label):
        jpg = QtGui.QPixmap(file).scaled(label.width(), label.height())
        label.setPixmap(jpg)
