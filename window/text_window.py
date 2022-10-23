from PyQt5 import uic

class Stats:
    def __init__(self):
        # 从文件中加载UI定义
        self.ui = uic.loadUi("ui/show_txt.ui")
        file = open('E:\Code\OpenCV_project\data\\result\chapter7\keypoint.txt', encoding='utf-8')
        keypoint = file.read()
        self.ui.label.setText(keypoint)