import os

from PyQt5 import uic
from PyQt5.QtWidgets import QFileDialog
from pyqt5_plugins.examplebuttonplugin import QtGui

import globalVar


class Stats:
    def __init__(self):
        # 从文件中加载UI定义
        self.ui = uic.loadUi("ui/answer_sheet.ui")

        # 路径初始化
        self.curPath = os.path.abspath(os.path.dirname(__file__))
        self.rootPath = self.curPath[:self.curPath.find("OpenCV_project\\") + len("OpenCV_project\\")]

        self.result = self.rootPath + '/data/result/'
        globalVar.set_value('result', self.result)

        # 绑定信号与槽
        self.ui.btn_ansload.clicked.connect(self.ansload)
        self.ui.btn_correct.clicked.connect(self.correct)

    def ansload(self):
        f = QFileDialog.getOpenFileName(None, "请选择读取的文件",
                                        self.rootPath, "JPG(*.jpg);;PNG(*.png);;MP4(*.mp4)")

        self.file = "".join(list(f[0]))
        globalVar.set_value('file', self.file)
        self.imshow(self.file, self.ui.img_origin)

    def correct(self):


    # 定义图片显示函数
    def imshow(self, file, label):
        jpg = QtGui.QPixmap(file).scaled(label.width(), label.height())
        label.setPixmap(jpg)
