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

        # slider初始化
        self.ui.sld_bright.setMaximum(100)  # 设置最大最小值
        self.ui.sld_bright.setMinimum(-100)
        self.ui.sld_bright.setSingleStep(1)  # 设置单步值
        self.ui.sld_bright.setValue(0)  # 设置初始值
        self.ui.sld_bright.setTickPosition(QSlider.TicksBelow)  # 设置刻度线位置

        # 绑定信号与槽
        self.ui.btn_load.clicked.connect(self.btn_load_read)
        self.ui.cbx_color.currentIndexChanged.connect(self.cbx_color_change)
        self.ui.cbx_edge.currentIndexChanged.connect(self.cbx_edge_change)
        self.ui.btn_merge.clicked.connect(self.btn_merge_click)
        self.ui.btn_zoom.clicked.connect(self.btn_zoom_click)
        self.ui.sld_bright.valueChanged.connect(self.brightness)

        # 路径初始化
        self.curPath = os.path.abspath(os.path.dirname(__file__))
        self.rootPath = self.curPath[:self.curPath.find("OpenCV_project\\") + len("OpenCV_project\\")]

        self.result = self.rootPath + '/data/result/'
        self.category = 'chapter1/'
        globalVar.set_value('result', self.result)

    # 页面切换
    def display1(self):
        self.ui.stackedWidget.setCurrentIndex(0)

    def display2(self):
        self.ui.stackedWidget.setCurrentIndex(1)

    # 动作函数
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
            cv2.imwrite(self.result + self.category + 'Blue.jpg', b)
            self.imshow(self.result + self.category + 'Blue.jpg', self.ui.img_new)
        elif method == 'Red':
            cv2.imwrite(self.result + self.category + 'Red.jpg', r)
            self.imshow(self.result + self.category + 'Red.jpg', self.ui.img_new)
        elif method == 'Green':
            cv2.imwrite(self.result + self.category + 'Green.jpg', g)
            self.imshow(self.result + self.category + 'Green.jpg', self.ui.img_new)

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
            cv2.imwrite(self.result + self.category + 'replicate.jpg', replicate)
            self.imshow(self.result + self.category + 'replicate.jpg', self.ui.img_new)
        elif method == '反射法':
            cv2.imwrite(self.result + self.category + 'reflect.jpg', reflect)
            self.imshow(self.result + self.category + 'reflect.jpg', self.ui.img_new)
        elif method == '反射法101':
            cv2.imwrite(self.result + self.category + 'reflect101.jpg', reflect101)
            self.imshow(self.result + self.category + 'reflect101.jpg', self.ui.img_new)
        elif method == '外包装法':
            cv2.imwrite(self.result + self.category + 'wrap.jpg', wrap)
            self.imshow(self.result + self.category + 'wrap.jpg', self.ui.img_new)
        elif method == '常量法':
            cv2.imwrite(self.result + self.category + 'constant.jpg', constant)
            self.imshow(self.result + self.category + 'constant.jpg', self.ui.img_new)

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
        cv2.imwrite(self.result + self.category + 'merge.jpg', res)
        self.imshow(self.result + self.category + 'merge.jpg', self.ui.img_new)

    # 图片缩放
    def btn_zoom_click(self):
        img = cv2.imread(self.file)
        fx = float(self.ui.txt_fx.text())
        fy = float(self.ui.txt_fy.text())
        res = cv2.resize(img, (0, 0), fx=fx, fy=fy)
        cv2.imwrite(self.result + self.category + 'zoom.jpg', res)
        self.imshow(self.result + self.category + 'zoom.jpg', self.ui.img_new)

    # 亮度调节
    def brightness(self):
        img = cv2.imread(self.file)
        level = self.ui.sld_bright.value()
        self.ui.lbl_bri.setText(str(level))
        # 创建一个与原图像一样大小的空白图像
        blank = np.zeros_like(img)
        if level >= 0:
            # 将原图像和空白图像相加即可增加亮度
            blank[:, :] = (level, level, level) # 空白图像的bgr都为50，这里增加或者减小值
            result = cv2.add(img, blank)
        else:
            # 将原图像和空白图像相减即可减小亮度
            blank[:, :] = (-level, -level, -level)
            result = cv2.subtract(img, blank)
        cv2.imwrite(self.result + self.category + 'brightness.jpg', result)
        self.imshow(self.result + self.category + 'brightness.jpg', self.ui.img_new)

    # 定义图片显示函数
    def imshow(self, file, label):
        jpg = QtGui.QPixmap(file).scaled(label.width(), label.height())
        label.setPixmap(jpg)
