import numpy as np
import sys
import os
import matplotlib
import time
import pyaudio
import pyqtgraph
from PyQt5.QtCore import Qt, QRegExp
from PyQt5.QtGui import QRegExpValidator, QFont
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QFileDialog, QMessageBox
from scipy import signal
from ui.Ui_detect import Ui_MainWindow

import scipy.io.wavfile as wav
from scipy.signal import resample

matplotlib.use("Qt5Agg")  # 声明使用QT5
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from src.src_2_yolo.yolo_detect_adjust import parse_opt, main

matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文
matplotlib.rcParams['axes.unicode_minus'] = False  # 显示负号

class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(QMainWindow, self).__init__()
        self.setup_ui()  # 渲染画布
        self.image_scale = 1.0  # 图片缩放比例
        self.pan_start = None  # 拖动的起始位置
        #elf.nosave = True  # 是否自动保存检测结果。autosave实现的时候，初始化为True
        self.nosave = False  # autosave功能没有实现，默认都保存

    def setup_ui(self):
        self.setupUi(self)  # 渲染pyqt5界面
        self.setWindowTitle("水声脉冲信号侦获软件")  # 设置窗口标题
        self.input_value_type()  # 输入值类型设置
        self.default_value()  # 默认值
        self.set_graph_ui()  # 设置画布
        self.connect_signals()  # 设置触发事件响应
        

    def input_value_type(self):
        '''
        输入值类型设置
        :return:
        '''
        reg_f = QRegExp('(-?\d*\.\d+|\d+)')  # 浮点数正则
        self.validator_f = QRegExpValidator(self)
        self.validator_f.setRegExp(reg_f)
        reg_i = QRegExp('(-?\d+)')  # 整数正则
        self.validator_i = QRegExpValidator(self)
        self.validator_i.setRegExp(reg_i)
    
    def default_value(self):
        '''
        默认值
        :return:
        '''
        self.iouSpinBox.setValue(0.45)  # iou阈值
        self.confSpinBox.setValue(0.25)  # conf阈值

        self.stftfsSpinBox.setValue(2000)  # stftfs初始值
        self.wlenspinBox.setValue(512)  # wlenspinBox初始值

        self.model_path.setText('未选择model文件') 
        self.inputwavpath.setText('未选择wav文件')
        self.output_path.setText('未选择文件保存地址')
    
    def set_graph_ui(self):
        '''
        设置画布
        :return:
        '''
        # 创建第1个画布和图形 (raw_img)
        self.raw_fig = plt.Figure()
        self.raw_canvas = FigureCanvas(self.raw_fig)
        raw_layout = QVBoxLayout()  # 垂直布局
        raw_layout.addWidget(self.raw_canvas)
        self.raw_img.setLayout(raw_layout)  # 设置布局到 raw_img

        # 创建第2个画布和图形 (detect_img)
        self.detect_fig = plt.Figure()
        self.detect_canvas = FigureCanvas(self.detect_fig)
        detect_layout = QVBoxLayout()  # 垂直布局
        detect_layout.addWidget(self.detect_canvas)
        self.detect_img.setLayout(detect_layout)  # 设置布局到 detect_img

    def connect_signals(self):
        '''
        设置触发事件响应
        :return:
        '''
        self.model_choose.clicked.connect(self.btn_modelchoose_clicked)  # 选择model文件

        self.wav_choose.clicked.connect(self.btn_wavchoose_clicked)  # 选择wav文件
        self.img_choose.clicked.connect(self.btn_imgchoose_clicked)  # 选择待检测的一张时频图
        self.imgs_choose.clicked.connect(self.btn_imgs_choose_clicked)  # 选择待检测的多张时频图
        self.realtime_chose.clicked.connect(self.btn_realtime_choose_clicked)  # 实时检测

        self.autosave_checkBox.stateChanged.connect(self.btn_autosave_checkBox_stateChanged)  # 自动保存检测结果

        self.outputpath_choose.clicked.connect(self.btn_outputpath_choose_clicked)  # 选择输出文件保存地址

        self.start.clicked.connect(self.btn_start_clicked)  # 开始检测
        self.stop.clicked.connect(self.btn_stop_clicked)  # 停止检测
        self.save.clicked.connect(self.btn_save_clicked)  # 保存检测结果

    def btn_realtime_choose_clicked(self):
        '''
        实时检测
        :return:
        '''
        self.detect_type = 'realtime'
        self.p = pyaudio.PyAudio()
        self.stream = None
        self.CHUNK = 1024  # 每次读取的音频数据块大小
        self.RATE = 8000  # 采样率
        self.fs = self.RATE  # STFT的采样率
        self.wlen = 512  # STFT的窗口长度
        self.mynoverlap = int(self.wlen * 0.75)  # 重叠长度
        # 初始化音频流
        if self.stream is None:
            self.stream = self.p.open(format=pyaudio.paInt16,
                                     channels=1,
                                     rate=self.RATE,
                                     input=True,
                                     frames_per_buffer=self.CHUNK)
        QMessageBox.information(self, u"Info", u"实时检测已启动", buttons=QMessageBox.Ok, defaultButton=QMessageBox.Ok)

    def btn_stop_clicked(self):
        '''
        停止检测
        :return:
        '''
        if self.timer:
            self.killTimer(self.timer)  # 停止定时器
            self.timer = None
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
                self.stream = None
            QMessageBox.information(self, u"Info", u"实时检测已停止", buttons=QMessageBox.Ok, defaultButton=QMessageBox.Ok)

    def btn_imgs_choose_clicked(self):
        '''
        选择待检测的多张时频图
        :return:
        '''
        self.detect_type = 'imgs'
        try:
            # 获取文件夹路径
            self.folder_path = QFileDialog.getExistingDirectory(self, "选择图片文件夹", "")

            # 如果用户选择了文件夹
            if self.folder_path:
                # 获取文件夹中的所有图片文件
                self.imgschoose_names = [os.path.join(self.folder_path, f) for f in os.listdir(self.folder_path) 
                            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
                
                if not self.imgschoose_names:
                    QMessageBox.warning(self, u"Warning", u"文件夹中没有图片文件", buttons=QMessageBox.Ok, defaultButton=QMessageBox.Ok)
                    return

        except Exception as e:
            QMessageBox.warning(self, u"Warning", f"检测过程中出现错误：{str(e)}", buttons=QMessageBox.Ok, defaultButton=QMessageBox.Ok)

    def btn_autosave_checkBox_stateChanged(self):
        '''
        自动保存检测结果
        :return:
        '''
        if self.autosave_checkBox.isChecked():
            self.nosave = False
        else:
            #self.nosave = True
            self.nosave = False #autosave功能没有实现，默认都保存

    def btn_outputpath_choose_clicked(self):
        '''
        选择输出文件保存地址
        :return:
        '''
        #获取输出文件保存地址
        save_dir = QFileDialog.getExistingDirectory(self, "选择文件夹", rf"D:\AAAprogramfile\vscode\pyqt\wav_Read_Plot\output")
        self.output_path.setText(save_dir)

    def btn_modelchoose_clicked(self):
        '''
        选择model文件
        :return:
        '''
        #获取model文件路径
        filepath, _ = QFileDialog.getOpenFileName(self, "选择model文件", rf"D:\AAAprogramfile\vscode\pyqt\wav_Read_Plot\weights", "Model Files (*.pt)")
        self.model_path.setText(filepath)

    def btn_wavchoose_clicked(self):
        '''
        选择wav文件
        :return:
        '''
        self.detect_type = 'wav'
        #获取wav文件路径
        filepath, _ = QFileDialog.getOpenFileName(self, "选择wav文件", "", "WAV Files (*.wav)")
        if not filepath:
            QMessageBox.warning(self, u"Warning", u"请先选择要检测的wav文件", buttons=QMessageBox.Ok, defaultButton=QMessageBox.Ok)
            return
        self.inputwavpath.setText(filepath)

    def btn_imgchoose_clicked(self):
        '''
        选择待检测的一张时频图
        :return:
        '''
        self.detect_type = 'img'
        try:
            self.imgchoose_name, _ = QFileDialog.getOpenFileName(self, "打开图片", "",
                                                           "*.png;;*.jpg;;All Files(*)")
        except OSError as reason:
            print('文件打开出错啊！核对路径是否正确' + str(reason))
        else:
            # 判断图片是否为空
            if not self.imgchoose_name:
                QMessageBox.warning(self, u"Warning", u"打开图片失败", buttons=QMessageBox.Ok,
                                              defaultButton=QMessageBox.Ok)
            else:
                self.display_raw_image(self.imgchoose_name)

    def display_raw_image(self, image_path):
        '''
        在raw_img中显示图片
        :param image_path: 图片路径
        :return:
        '''
        self.raw_fig.clear()  # 清除之前的图像
        ax = self.raw_fig.add_subplot(111)
        img = mpimg.imread(image_path)
        ax.imshow(img)
        ax.axis('off')  # 不显示坐标轴
        self.raw_canvas.draw()  # 重新绘制画布

        # 连接鼠标事件
        self.raw_canvas.mpl_connect("motion_notify_event", self.on_mouse_move)
#        self.raw_canvas.mpl_connect("scroll_event", self.on_mouse_scroll)
#        self.raw_canvas.mpl_connect("button_press_event", self.on_mouse_press)
#        self.raw_canvas.mpl_connect("button_release_event", self.on_mouse_release)

    def on_mouse_move(self, event):
        '''
        鼠标移动事件：显示光标位置
        :param event: 鼠标事件
        :return:
        '''
        if event.inaxes is not None:
            x, y = int(event.xdata), int(event.ydata)
            self.result_listWidget.clear()  # 清空之前的内容
            self.result_listWidget.addItem(f"光标位置: ({x}, {y})")  # 添加最新位置
#        pass

    def on_mouse_scroll(self, event):
        '''
        鼠标滚轮事件：缩放图片
        :param event: 鼠标滚轮事件
        :return:
        '''
#        if event.inaxes is not None:
#            scale_factor = 1.1 if event.button == 'up' else 0.9
#            self.image_scale *= scale_factor
#            ax = self.raw_fig.gca()
#            ax.set_xlim(ax.get_xlim()[0] * scale_factor, ax.get_xlim()[1] * scale_factor)
#            ax.set_ylim(ax.get_ylim()[0] * scale_factor, ax.get_ylim()[1] * scale_factor)
#            self.raw_canvas.draw()
        pass

    def on_mouse_press(self, event):
        '''
        鼠标按下事件：开始拖动
        :param event: 鼠标按下事件
        :return:
        '''
#        if event.button == 1:  # 左键
#            self.pan_start = (event.xdata, event.ydata)
        pass

    def on_mouse_release(self, event):
        '''
        鼠标释放事件：结束拖动
        :param event: 鼠标释放事件
        :return:
        '''
#        if event.button == 1 and self.pan_start is not None:  # 左键
#            x_start, y_start = self.pan_start
#            x_end, y_end = event.xdata, event.ydata
#            ax = self.raw_fig.gca()
#            ax.set_xlim(ax.get_xlim()[0] + (x_start - x_end), ax.set_ylim(ax.get_ylim()[0] + (y_start - y_end)))
#            self.raw_canvas.draw()
#            self.pan_start = None
        pass
    
    def btn_start_clicked(self):
        '''
        开始检测
        :return:
        '''
        # 构造参数，设置参数值
        weights_path = self.model_path.text()
        save_dir = self.output_path.text()
        iou = self.iouSpinBox.value()
        conf = self.confSpinBox.value()
        wav_path = self.inputwavpath.text()

        fs = self.stftfsSpinBox.value()
        wlen = self.wlenspinBox.value()
        mynoverlap = int(wlen * 0.75)

        opt = parse_opt()
        opt.weights = weights_path
        opt.iou_thres = iou
        opt.conf_thres = conf
        opt.project = save_dir
        opt.nosave = self.nosave

        if self.detect_type == 'img':
            try:
                img_of_img = self.imgchoose_name.replace('/', '\\')
                opt.source = img_of_img
                save_img = main(opt)
                print(f"检测结果已保存至：{save_img}")
                self.display_detect_image(save_img)
            except:
                QMessageBox.warning(self, u"Warning", u"请先选择要检测的图片文件",
                                            buttons=QMessageBox.Ok,
                                            defaultButton=QMessageBox.Ok)
                return
        elif self.detect_type == 'imgs':
            try:
                for img_of_imgs in self.imgschoose_names:
                    opt.source = img_of_imgs
                    save_img = main(opt)
                    print(f"检测结果已保存至：{save_img}")
                # 显示第一张图片的检测结果
                if self.imgschoose_names:
                    self.display_detect_image(save_img)
                    QMessageBox.information(self, u"Info", u"所有图片检测完成", buttons=QMessageBox.Ok, defaultButton=QMessageBox.Ok)
            except:
                QMessageBox.warning(self, u"Warning", u"请先选择要检测的图片文件夹",
                                            buttons=QMessageBox.Ok,
                                            defaultButton=QMessageBox.Ok)
                return
        elif self.detect_type == 'wav':
            try:
                wav_fs, wav_data = wav.read(wav_path)
                # 如果.wav文件是立体声，只取其中一个声道
                if len(wav_data.shape) > 1:
                    wav_data = wav_data[:, 0]  # 取左声道
                # 对信号进行重采样
                if wav_fs != fs:
                    # 计算重采样后的长度
                    resample_length = int(len(wav_data) * fs / wav_fs)
                    wav_data = resample(wav_data, resample_length)  # 重采样
                    wav_fs = fs  # 更新采样频率
                f, t, spectrum = signal.stft(wav_data, fs=fs, nperseg=wlen, noverlap=mynoverlap)
                # 对spectrum进行归一化处理
                spectrum = spectrum / np.max(np.abs(spectrum))

                # 设置绘制的时频图大小
                width_pixels = 640  # 图像宽度（像素）
                height_pixels = 480  # 图像高度（像素）
                dpi = 100  # 每英寸点数
                fig_width = width_pixels / dpi  # 转换为英寸
                fig_height = height_pixels / dpi  # 转换为英寸
                # 设置画布大小
                self.raw_fig.set_size_inches(fig_width, fig_height)
                # 绘制时频图
                self.raw_fig.clear()
                ax = self.raw_fig.add_subplot(111)
                #self.raw_fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)  # 调整布局的位置，可以省去
                ax.pcolormesh(t, f, np.abs(spectrum), shading='gouraud')  # 绘制时频图，vmin, vmax
                ax.set_xlabel('Time [s]')
                ax.set_ylabel('Frequency [Hz]')
                ax.set_title('STFT Magnitude')
                self.raw_canvas.draw()

                # 保存时频图
                if not os.path.exists(rf".\result\wav_tempsav"):
                    os.makedirs(rf".\result\wav_tempsav")
                stft_save_path = os.path.join(rf".\result\wav_tempsav", "stft_temp.png")
                self.raw_fig.savefig(stft_save_path, dpi=dpi, bbox_inches='tight')
                print(f"时频图已保存至：{stft_save_path}")

                # 调用YOLO检测模型
                opt.source = stft_save_path
                save_img = main(opt)
                print(f"检测结果已保存至：{save_img}")
                # 显示检测结果
                self.display_detect_image(save_img)
            except Exception as e:
                QMessageBox.warning(self, u"Warning", f"处理wav文件时出错：{str(e)}",
                                            buttons=QMessageBox.Ok,
                                            defaultButton=QMessageBox.Ok)
                return         
        elif self.detect_type == 'realtime':
            try:
                self.timer = None  # 定时器
                self.waterfall_data = None  # 瀑布图数据
                self.time_axis = None  # 时间轴
                self.freq_axis = None  # 频率轴
                self.current_time_index = 0  # 当前时间索引
                self.start_time = None  # 记录程序开始的时间
                self.colorbar = None  # 颜色条对象
                # 初始化瀑布图
                self.raw_fig.clear()
                self.ax = self.raw_fig.add_subplot(111)
                self.ax.set_xlabel('Frequency [Hz]')
                self.ax.set_ylabel('Time [s]')
                self.ax.set_title('Real-time STFT Waterfall')
                # 初始化时间轴和频率轴
                self.time_axis = np.linspace(0, 10, 100)  # 10秒的时间轴
                self.freq_axis = np.linspace(0, self.fs / 2, self.wlen // 2 + 1)
                # 初始化瀑布图数据（设置为一个非常小的值，避免dBFs计算为负无穷）
                self.waterfall_data = np.full((len(self.time_axis), len(self.freq_axis)), -140)
                # 初始化颜色条
                im = self.ax.pcolormesh(self.freq_axis, self.time_axis, self.waterfall_data, shading='gouraud', cmap='viridis')
                self.colorbar = self.raw_fig.colorbar(im, ax=self.ax, label='Magnitude (dBFs)')  # 创建颜色条
                # 启动定时器
                self.timer = self.startTimer(100)  # 每100ms刷新一次
            except Exception as e:
                QMessageBox.warning(self, u"Warning", f"初始化实时检测时出错：{str(e)}", buttons=QMessageBox.Ok, defaultButton=QMessageBox.Ok)
                
    def timerEvent(self, event):
        '''
        定时器事件：更新实时时频瀑布图
        :param event: 定时器事件
        :return:
        '''
        if self.detect_type == 'realtime' and self.timer and self.stream:
            try:
                # 从麦克风读取音频数据
                data = self.stream.read(self.CHUNK, exception_on_overflow=False)
                audio_data = np.frombuffer(data, dtype=np.int16)
                # 计算STFT
                _, _, spectrum = signal.stft(audio_data, fs=self.fs, nperseg=self.wlen, noverlap=self.mynoverlap)
                spectrum = np.abs(spectrum)  # 取幅度值
                # 将spectrum转换为dBFs
                spectrum = 20 * np.log10(np.maximum(spectrum / 32767, 1e-10))  # PaInt16使用满幅32767作为参考电平
                # 更新瀑布图数据
                self.waterfall_data = np.roll(self.waterfall_data, -1, axis=0)
                self.waterfall_data[-1, :] = np.mean(spectrum, axis=1)
                # 更新时间轴
                current_time = time.time() - self.start_time if self.start_time else 0
                self.time_axis = np.linspace(current_time, current_time + 10, len(self.time_axis))  # 时间轴始终显示10秒的范围
                # 绘制瀑布图
                self.ax.clear()
                im = self.ax.pcolormesh(self.freq_axis, self.time_axis, self.waterfall_data, shading='gouraud', cmap='viridis')
                self.ax.set_xlabel('Frequency [Hz]')
                self.ax.set_ylabel('Time [s]')
                self.ax.set_title('Real-time STFT Waterfall (dBFs)')
                self.colorbar.update_normal(im)
                self.raw_canvas.draw()
            except Exception as e:
                QMessageBox.warning(self, u"Warning", f"更新实时检测时出错：{str(e)}", buttons=QMessageBox.Ok, defaultButton=QMessageBox.Ok)
                self.killTimer(self.timer)
                self.timer = None

    def display_detect_image(self, image_path):
        '''
        在detect_img中显示图片
        :param image_path: 图片路径
        :return:
        '''
        self.detect_fig.clear()  # 清除之前的图像
        ax = self.detect_fig.add_subplot(111)
        img = mpimg.imread(image_path)
        ax.imshow(img)
        ax.axis('off')  # 不显示坐标轴
        self.detect_canvas.draw()  # 重新绘制画布

    def btn_save_clicked(self):
        '''
        点击save按钮，保存画布上的图片至指定文件夹
        :return:
        '''
        # 打开文件对话框，选择保存路径
        filepath, _ = QFileDialog.getSaveFileName(
            self, "保存图片", rf"D:\AAAprogramfile\vscode\pyqt\wav_Read_Plot\result", "PNG Files (*.png);;JPEG Files (*.jpg);;All Files (*)"
        )

        # 如果用户选择了保存路径
        if filepath:
            try:
                # 保存当前画布上的图像
                self.detect_canvas.figure.savefig(filepath, dpi=300, bbox_inches='tight')
                print(f"图片已保存至：{filepath}")
            except Exception as e:
                print(f"保存图片失败：{e}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    mywindow = MainWindow()
    mywindow.show()
    sys.exit(app.exec_())