import os

import cv2
from PyQt5 import uic
from PyQt5.QtWidgets import QFileDialog
from pyqt5_plugins.examplebuttonplugin import QtGui

import globalVar


class Stats:
    def __init__(self):
        # 从文件中加载UI定义
        self.ui = uic.loadUi("ui/panorama.ui")
        # 路径初始化
        self.curPath = os.path.abspath(os.path.dirname(__file__))
        self.rootPath = self.curPath[:self.curPath.find("OpenCV_project\\") + len("OpenCV_project\\")]

        self.result = self.rootPath + '/data/result/'
        globalVar.set_value('result', self.result)

        # 信号与槽
        self.ui.btn_open.clicked.connect(self.openFile)
        self.ui.btn_emerge.clicked.connect(self.emerge)

    # 打开文件
    def openFile(self):
        # 其中self指向自身，"读取文件夹"为标题名，"./"为打开时候的当前路径
        directory = QFileDialog.getExistingDirectory(self.ui, "选取文件夹", "./")  # 起始路径
        myFolders = directory.split("/")[-1]
        mainFolder = directory.split("/")[-2]
        globalVar.set_value('myFolders', myFolders)
        globalVar.set_value('mainFolder', mainFolder)

    def emerge(self):
        mainFolder = globalVar.get_value('mainFolder')
        myFolders = os.listdir(mainFolder)
        for folder in myFolders:
            path = mainFolder + '/' + folder
            images = []
            myList = os.listdir(path)
            print(f'Total number of images detected {len(myList)}')
            for imgN in myList:
                curImg = cv2.imread(f'{path}/{imgN}')
                curImg = cv2.resize(curImg, (0, 0), None, 0.2, 0.2)
                images.append(curImg)

            stitcher = cv2.Stitcher.create()
            (status, result) = stitcher.stitch(images)

            if status == cv2.STITCHER_OK:
                self.ui.lbl_status.setText('Panorama Generated')
                print(self.result)
                cv2.imwrite('E:\Code\OpenCV_project\data\\result\\Panorama result.jpg', result)
                self.imshow('E:\Code\OpenCV_project\data\\result\\Panorama result.jpg', self.ui.lbl_show)
            else:
                self.ui.lbl_status.setText('Panorama Generation Unsuccessful')

    def imshow(self, file, label):
        jpg = QtGui.QPixmap(file).scaled(label.width(), label.height())
        label.setPixmap(jpg)
