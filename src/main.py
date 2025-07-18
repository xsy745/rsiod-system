"""
"""
# @author: xsy745
# @createTime:


import sys
import yaml

from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QFileDialog, QGraphicsScene, QMessageBox

import main_win
from src.predictor import infer


def show_message(message_type, message_content):
    """显示提示框"""
    msg_box = QMessageBox()
    # 设置图标类型
    # if message_type == '错误':
    #     msg_box.setIcon(QMessageBox.NoIcon)
    # elif message_type == '成功':
    #     msg_box.setIcon(QMessageBox.Critical)
    msg_box.setIcon(QMessageBox.NoIcon)
    # 设置弹出框标题
    msg_box.setWindowTitle(message_type)
    # 设置信息
    msg_box.setText(message_content)
    # 显示弹出框
    msg_box.exec_()


class MainWindow(QtWidgets.QMainWindow, main_win.Ui_MainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setupUi(self)
        self.image = None
        self.scene_1 = QGraphicsScene(self)
        self.scene_2 = QGraphicsScene(self)
        self.graphicsView_1.setScene(self.scene_1)
        self.graphicsView_2.setScene(self.scene_2)
        self.pushButton_1.clicked.connect(self.set_image)
        self.pushButton_2.clicked.connect(self.detect)
        
        # 加载配置文件
        with open('./config.yml', 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)


    def set_image(self):
        """设置检测图像"""

        # 打开文件选择对话框，仅允许选择图片文件
        options = QFileDialog.Options()
        image_dir, _ = QFileDialog.getOpenFileName(
            # 父窗口
            self,
            # 文件选择对话框标题
            self.config['ui']['dialog_title'],
            # 默认打开路径
            '',
            # 图片文件过滤器
            self.config['ui']['image_filter'],
            options=options
        )
        if image_dir:
            try:
                self.display_image(1, image_dir)
                self.image = image_dir
                # print(self.image)
            except Exception as e:
                # show_message('错误', f'无法加载图片文件。详细信息：{e}')
                show_message('错误', self.config['messages']['load_error'].format(e))

    def display_image(self, view_type, image_dir=None):
        """在GUI中展示图片"""
        if view_type == 1:
            # 清空场景内容
            self.scene_1.clear()
            # 加载图片到 QPixmap
            pixmap = QPixmap(image_dir)

            # 检查图片是否加载成功
            if pixmap.isNull():
                raise ValueError('pixmap is null')

            # 获取 QGraphicsView 的可见区域大小
            view_width = self.graphicsView_1.width()
            view_height = self.graphicsView_1.height()

            # 缩放图片以适应窗口（保持宽高比）
            scaled_pixmap = pixmap.scaled(view_width, view_height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            item = QtWidgets.QGraphicsPixmapItem(scaled_pixmap)
            self.scene_1.addItem(item)

            # 自动调整 QGraphicsView 的缩放比例以适应图片
            self.graphicsView_1.fitInView(self.scene_1.sceneRect(), Qt.KeepAspectRatio)
            # or self.graphicsView_1.fitInView(item)
            
        elif view_type == 2:
            self.scene_2.clear()
            pixmap = QPixmap(image_dir)
            # print(image_dir)
            if pixmap.isNull():
                raise ValueError('pixmap is null')
            view_width = self.graphicsView_2.width()
            view_height = self.graphicsView_2.height()
            scaled_pixmap = pixmap.scaled(view_width, view_height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            item = QtWidgets.QGraphicsPixmapItem(scaled_pixmap)
            self.scene_2.addItem(item)
            self.graphicsView_2.fitInView(self.scene_2.sceneRect(), Qt.KeepAspectRatio)


    def detect(self):
        model_dir = self.config['model']['path']
        image_dir = self.image
        if self.image is None:
            show_message('错误', self.config['messages']['error_no_image'])
            return
        show_message('成功', self.config['messages']['start_detect'])
        detected_image_dir, object_info = infer(model_dir, image_dir)
        show_message('成功', self.config['messages']['detect_complete'].format(detected_image_dir))
        try:
            self.display_image(2, detected_image_dir)
            self.textBrowser.setText(object_info)
        except Exception as e:
            show_message('错误', f'无法加载图片文件。详细信息：{e}')


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())

