import os
import sys

import PySide2
from PyQt5.QtWidgets import QApplication

import globalVar
from window import main_window

if __name__ == "__main__":
    # 创建环境变量
    dirname = os.path.dirname(PySide2.__file__)
    plugin_path = os.path.join(dirname, 'plugins', 'platforms')
    os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = plugin_path
    app = QApplication(sys.argv)

    # 样式表
    # apply_stylesheet(app, theme='light_lightgreen.xml')

    # 环境变量初始化
    globalVar.init()

    # 打开登录窗口
    window = main_window.Stats()
    window.ui.show()
    sys.exit(app.exec_())
