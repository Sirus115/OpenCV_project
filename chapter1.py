import os

from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QPlainTextEdit, QMessageBox, QLabel, QFileDialog
from PyQt5 import uic
from PyQt5.QtGui import QIcon, QImage, QPixmap
import cv2  # opencv读取的格式是BGR
import matplotlib.pyplot as plt
import numpy as np
from PySide2.QtUiTools import QUiLoader
from pyqt5_plugins.examplebuttonplugin import QtGui

import globalVar


class chapter1:
    def __init__(self):
        self.ui = uic.loadUi("ui/main.ui")
        # 绑定信号与槽
        self.ui.btn_load.clicked.connect(self.btn_load_read)
        self.ui.cbx_color.currentIndexChanged.connect(self.cbx_color_change)

    # 动作函数
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

    def cbx_color_change(self):
        img = cv2.imread(self.file)
        b, g, r = cv2.split(img)
        method = self.ui.cbx_color.currentText()
        # print(self.result + self.category)
        if method == 'Blue':
            cv2.imwrite(self.result + self.category + 'Blue.jpg', b)
            self.imshow(self.result + self.category + 'Blue.jpg', self.ui.img_new)
        elif method == 'Red':
            cv2.imwrite(self.result + self.category + 'Red.jpg', r)
            self.imshow(self.result + self.category + 'Red.jpg', self.ui.img_new)
        elif method == 'Green':
            cv2.imwrite(self.result + self.category + 'Green.jpg', g)
            self.imshow(self.result + self.category + 'Green.jpg', self.ui.img_new)
