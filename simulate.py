import numpy as np
import sys
import matplotlib
import pyqtgraph
from PyQt5.QtCore import Qt, QRegExp
from PyQt5.QtGui import QRegExpValidator, QFont
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QFileDialog
from scipy import signal
from ui.Ui_simulate import Ui_MainWindow

import scipy.io.wavfile as wav
from scipy.signal import resample

matplotlib.use("Qt5Agg")  # 声明使用QT5
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt

matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文
matplotlib.rcParams['axes.unicode_minus'] = False  # 显示负号

class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(QMainWindow, self).__init__()
        self.setup_ui()  # 渲染画布
        
    ''' 
        #功能：min\close\max生效       
        # style 1: window can be stretched
        self.setWindowFlags(Qt.CustomizeWindowHint | Qt.WindowStaysOnTopHint)

        # style 2: window can not be stretched
        #self.setWindowFlags(Qt.Window | Qt.FramelessWindowHint
        #                    | Qt.WindowSystemMenuHint | Qt.WindowMinimizeButtonHint | Qt.WindowMaximizeButtonHint)
        # self.setWindowOpacity(0.85)  # Transparency of window

        self.minButton.clicked.connect(self.showMinimized)
        self.maxButton.clicked.connect(self.max_or_restore)
        # show Maximized window
        self.maxButton.animateClick(10)
        self.closeButton.clicked.connect(self.close)
    
    def max_or_restore(self):
        if self.isMaximized():
            self.showNormal()
            self.maxButton.setText("最大化")
        else:
            self.showMaximized()
            self.maxButton.setText("还原")
    '''
    def setup_ui(self):
        self.setupUi(self)  # 渲染pyqt5界面
        self.setWindowTitle("水声脉冲信号仿真软件")
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

        ## 设置输入框只能输入整数
        #self.signalfsSpinBox.setValidator(self.validator_i)
        #self.signalNPSpinBox.setValidator(self.validator_i)
        #self.signalNSpinBox.setValidator(self.validator_i)
        #self.sig1f0.setValidator(self.validator_i)
        #self.sig1f1.setValidator(self.validator_i)
        #self.sig2f0.setValidator(self.validator_i)
        #self.sig2f1.setValidator(self.validator_i)
        #self.sig3f0.setValidator(self.validator_i)
        #self.sig3f1.setValidator(self.validator_i)
        #self.snrSpinBox.setValidator(self.validator_i)
        #self.stftfsSpinBox.setValidator(self.validator_i)
        #self.wlenspinBox.setValidator(self.validator_i)

        ## 设置输入框只能输入浮点数
        #self.signalTSpinBox.setValidator(self.validator_f)
        #self.sig1t0.setValidator(self.validator_f)
        #self.sig1t1.setValidator(self.validator_f)
        #self.sig2t0.setValidator(self.validator_f)
        #self.sig2t1.setValidator(self.validator_f)
        #self.sig3t0.setValidator(self.validator_f)
        #self.sig3t1.setValidator(self.validator_f)
        #self.wavdurationSpinBox.setValidator(self.validator_f)
        #self.wavstarttimeSpinBox.setValidator(self.validator_f)        
        
    def default_value(self):
        '''
        默认值
        :return:
        '''
        self.is_analysis = False
        #signal仿真参数
        self.signalfsSpinBox.setValue(2000)  #仿真脉冲信号的采样频率fs
        self.signalTSpinBox.setValue(5.00)  #仿真脉冲信号的单周期长度T
        self.signalNPSpinBox.setValue(1)  #仿真脉冲信号的周期数NP
        '''
        signal的下拉框里一共有四种选项,分别是NO、CW、LFM、HFM
        默认为NO,即不仿真脉冲信号
        '''
        #sig1参数
        self.sig1.setCurrentIndex(self.sig1.findText('NO'))  #设置默认值
        self.sig1f0.setValue(0)  #sig1的起始频率f0
        self.sig1f1.setValue(0)  #sig1的终止频率f1
        self.sig1t0.setValue(0.00)  #sig1的起始时间t0
        self.sig1t1.setValue(0.00)  #sig1的终止时间t1

        #sig2参数
        self.sig2.setCurrentIndex(self.sig2.findText('NO'))  #设置默认值
        self.sig2f0.setValue(0)  #sig2的起始频率f0
        self.sig2f1.setValue(0)  #sig2的终止频率f1
        self.sig2t0.setValue(0.00)  #sig2的起始时间t0
        self.sig2t1.setValue(0.00)  #sig2的终止时间t1

        #sig3参数
        self.sig3.setCurrentIndex(self.sig3.findText('NO'))  #设置默认值
        self.sig3f0.setValue(0)  #sig3的起始频率f0
        self.sig3f1.setValue(0)  #sig3的终止频率f1
        self.sig3t0.setValue(0.00)  #sig3的起始时间t0
        self.sig3t1.setValue(0.00)  #sig3的终止时间t1

        #wav文件参数
        '''
        wav一栏中可以选择是否添加背景噪声
        '''
        self.wavdurationSpinBox.setValue(0.00)  #wav文件的持续时间
        self.wavstarttimeSpinBox.setValue(0.00)  #wav文件的开始时间
        self.wavfilepath.setText('未选择wav文件')

        #stft参数
        self.snrSpinBox.setValue(0)  #信噪比SNR
        self.stftfsSpinBox.setValue(2000)  #stft的采样频率fs
        self.wlenspinBox.setValue(256)  #stft的窗长

    def set_graph_ui(self):
        self.stft_fig = plt.Figure()
        self.stft_canvas = FigureCanvas(self.stft_fig)
        layout = QVBoxLayout()  # 垂直布局
        layout.addWidget(self.stft_canvas)
        self.stft_show.setLayout(layout)  # 设置好布局之后调用函数

    def connect_signals(self):
        '''
        按钮，输入框触发事件
        :return:
        '''
        self.wavchoose.clicked.connect(self.btn_wavchoose_clicked)  #选择wav文件,并显示
        self.start.clicked.connect(self.btn_start_clicked)  # 点击start按钮触发
        self.save.clicked.connect(self.btn_save_clicked)  # 点击save按钮触发
    
    def btn_wavchoose_clicked(self):
        '''
        选择wav文件
        :return:
        '''
        #获取wav文件路径
        filepath, _ = QFileDialog.getOpenFileName(self, "选择wav文件", "", "WAV Files (*.wav)")
        self.wavfilepath.setText(filepath)

    def btn_start_clicked(self):
        '''
        点击start按钮触发
        :return:
        '''
        #sig1
        signal1 = self.sig1.currentText()
        if signal1 == 'NO':
            t1, samples1 = self.get_NO_period()
        elif signal1 == 'CW':
            t1, samples1 = self.get_CW_period()
        elif signal1 == 'LFM':
            t1, samples1 = self.get_LFM_period()
        elif signal1 == 'HFM':
            t1, samples1 = self.get_HFM_period()

        #sig2
        signal2 = self.sig2.currentText()
        if signal2 == 'NO':
            t2, samples2 = self.get_NO_period(signal_type='2')
        elif signal2 == 'CW':
            t2, samples2 = self.get_CW_period(signal_type='2')
        elif signal2 == 'LFM':
            t2, samples2 = self.get_LFM_period(signal_type='2')
        elif signal2 == 'HFM':
            t2, samples2 = self.get_HFM_period(signal_type='2')

        #sig3
        signal3 = self.sig3.currentText()
        if signal3 == 'NO':
            t3, samples3 = self.get_NO_period(signal_type='3')
        elif signal3 == 'CW':
            t3, samples3 = self.get_CW_period(signal_type='3')
        elif signal3 == 'LFM':
            t3, samples3 = self.get_LFM_period(signal_type='3')
        elif signal3 == 'HFM':
            t3, samples3 = self.get_HFM_period(signal_type='3')

        #混合信号
        self.mix_samples = samples1 + samples2 + samples3  # 混合纯仿真信号
        self.t = t1  # t= t1 = t2 = t3
        self.is_add_noise = False

        #初始化self.sig, self.noise, self.noisy_sig
        self.sig = self.mix_samples  # 纯信号
        self.noise = np.zeros_like(self.mix_samples)
        self.noisy_sig = self.mix_samples

        # 周期数 NP
        NP = self.signalNPSpinBox.value()  # 获取周期数
        if NP > 1:
            self.mix_samples = np.tile(self.mix_samples, NP)  # 将混合信号重复 NP 次
            self.sig = np.tile(self.sig, NP)  # 将纯信号重复 NP 次
            self.noise = np.tile(self.noise, NP)  # 将噪声重复 NP 次
            self.noisy_sig = np.tile(self.noisy_sig, NP)  # 将含噪信号重复 NP 次
            self.t = np.tile(self.t, NP)  # 将时间轴重复 NP 次

        # 添加噪声
        if self.wavfilepath.text() != '未选择wav文件':
            self.sig, self.noise, self.noisy_sig = self.add_noise()  # 叠加.wav文件噪声

        # 绘制stft图
        stft_choose = self.stftcomboBox.currentText()
        if self.is_add_noise == False:
            self.plot_fig(self.mix_samples)
        else: 
            if stft_choose == 'sig':
                self.plot_fig(self.sig)
            elif stft_choose == 'noise':
                self.plot_fig(self.noise)
            elif stft_choose == 'noisy_sig':
                self.plot_fig(self.noisy_sig)

    def plot_fig(self, data):
        '''
        绘制信号图
        :param data: 信号数据
        :return:
        '''
        self.stft_fig.clear()  # 清空画布
        fig_choose = self.plotcomboBox.currentText()
        if fig_choose == 'stft':
            self.plot_stft(data)
        elif fig_choose == 'fft':
            self.plot_fft(data)
        elif fig_choose == 'wav':
            self.plot_time(data)

    def plot_stft(self, data):
        '''
        绘制stft图
        :param data: 信号数据
        :return:
        '''
        mynoverlap = self.wlenspinBox.value()*0.75
        f, t, spectrum = signal.stft(data, fs=self.stftfsSpinBox.value(),
                                nperseg=self.wlenspinBox.value(), noverlap=mynoverlap)
        
        self.stft_canvas.figure.clear()  # 清空画布
        ax = self.stft_fig.add_subplot(111)
        self.stft_fig.subplots_adjust(left=None, bottom=0.2, right=None, top=None, wspace=None, hspace=None)
        ax.cla()  # 删除原图，让画布上只有新的一次的图
        ax.pcolormesh(t, f, np.abs(spectrum), vmin=0, vmax=0.1, shading='gouraud')
        ax.set_title('STFT', fontsize=20)
        ax.set_xlabel('time [sec]', fontsize=20)
        ax.set_ylabel('frequency [Hz]', fontsize=20)
        self.stft_canvas.draw()

    def plot_fft(self, data):
        '''
        绘制FFT图
        :param data: 信号数据
        :return:
        '''
        fs = self.stftfsSpinBox.value()  # 获取采样频率
        N = len(data)  # 信号长度
        fft_result = np.fft.fft(data)  # 计算FFT
        fft_freqs = np.fft.fftfreq(N, 1/fs)  # 计算频率轴

        # 取正频率部分
        positive_freqs = fft_freqs[:N//2]
        positive_fft = np.abs(fft_result[:N//2])

        # 清空画布并绘制FFT图
        self.stft_canvas.figure.clear()
        ax = self.stft_fig.add_subplot(111)
        ax.plot(positive_freqs, positive_fft)
        ax.set_title('FFT', fontsize=20)
        ax.set_xlabel('Frequency [Hz]', fontsize=20)
        ax.set_ylabel('Amplitude', fontsize=20)
        ax.grid(True)
        self.stft_canvas.draw()

    def plot_time(self, data):
        '''
        绘制时域波形图
        :param data: 信号数据
        :return:
        '''
        fs = self.stftfsSpinBox.value()  # 获取采样频率
        t = np.arange(len(data)) / fs  # 生成时间轴

        # 清空画布并绘制时域波形图
        self.stft_canvas.figure.clear()
        ax = self.stft_fig.add_subplot(111)
        ax.plot(t, data)
        ax.set_title('Time Domain Waveform', fontsize=20)
        ax.set_xlabel('Time [sec]', fontsize=20)
        ax.set_ylabel('Amplitude', fontsize=20)
        ax.grid(True)
        self.stft_canvas.draw()
        
    def get_NO_period(self, signal_type='1'):
        '''
        获取NO信号周期
        :return:
        ''' 
        fs = self.signalfsSpinBox.value()  #采样频率
        T = self.signalTSpinBox.value()  #单周期长度
        NP = self.signalNPSpinBox.value()  #周期数
        t = np.arange(0, T, 1/fs)  #时间序列
        samples_zero = np.zeros_like(t)  #信号序列

        if signal_type == '1':
            f0 = self.sig1f0.value()  #起始频率
            f1 = self.sig1f1.value()  #终止频率,cw中f1=f0
            t0 = self.sig1t0.value()  #起始时间
            t1 = self.sig1t1.value()  #终止时间
        elif signal_type == '2':
            f0 = self.sig2f0.value()
            f1 = self.sig2f1.value()
            t0 = self.sig2t0.value()
            t1 = self.sig2t1.value()
        elif signal_type == '3':
            f0 = self.sig3f0.value()
            f1 = self.sig3f1.value()
            t0 = self.sig3t0.value()
            t1 = self.sig3t1.value()

        return t, samples_zero
    
    def get_CW_period(self, signal_type='1'):
        '''
        获取CW信号周期
        :return:
        '''
        fs = self.signalfsSpinBox.value()  #采样频率
        T = self.signalTSpinBox.value()  #单周期长度
        NP = self.signalNPSpinBox.value()  #周期数
        t = np.arange(0, T, 1/fs)  #时间序列
        samples_zero = np.zeros_like(t)  #信号序列

        if signal_type == '1':
            f0 = self.sig1f0.value()  #起始频率
            f1 = self.sig1f1.value()  #终止频率,cw中f1=f0
            t0 = self.sig1t0.value()  #起始时间
            t1 = self.sig1t1.value()  #终止时间
        elif signal_type == '2':
            f0 = self.sig2f0.value()
            f1 = self.sig2f1.value()
            t0 = self.sig2t0.value()
            t1 = self.sig2t1.value()
        elif signal_type == '3':
            f0 = self.sig3f0.value()
            f1 = self.sig3f1.value()
            t0 = self.sig3t0.value()
            t1 = self.sig3t1.value()

        samples_cw = np.zeros_like(t)  #信号序列初始化
        pulse_mask= (t>=t0) & (t<=t1)  #脉冲信号的时间范围
        samples_cw[pulse_mask] = np.cos(2*np.pi*f0*t[pulse_mask])  #脉冲信号
        samples = samples_zero + samples_cw  #混合信号
        return t, samples
    
    def get_LFM_period(self, signal_type='1'):
        '''
        获取LFM信号周期
        :return:
        '''
        fs = self.signalfsSpinBox.value()  #采样频率
        T = self.signalTSpinBox.value()  #单周期长度
        NP = self.signalNPSpinBox.value()  #周期数
        t = np.arange(0, T, 1/fs)  #时间序列
        samples_zero = np.zeros_like(t)  #信号序列

        if signal_type == '1':
            f0 = self.sig1f0.value()  #起始频率
            f1 = self.sig1f1.value()  #终止频率
            t0 = self.sig1t0.value()  #起始时间
            t1 = self.sig1t1.value()  #终止时间
        elif signal_type == '2':
            f0 = self.sig2f0.value()
            f1 = self.sig2f1.value()
            t0 = self.sig2t0.value()
            t1 = self.sig2t1.value()
        elif signal_type == '3':
            f0 = self.sig3f0.value()
            f1 = self.sig3f1.value()
            t0 = self.sig3t0.value()
            t1 = self.sig3t1.value()
        
        samples_lfm = np.zeros_like(t)  #信号序列初始化
        pulse_mask= (t>=t0) & (t<=t1)  #脉冲信号的时间范围
        k=(f1-f0)/(t1-t0)  #斜率
        t_adjusted = t[pulse_mask] - t0  #调整时间轴，使其相对于 t0!!!
        samples_lfm[pulse_mask] = np.cos(2*np.pi * (f0 * t_adjusted + k / 2 * t_adjusted**2))  # 脉冲信号
        samples = samples_zero + samples_lfm  #混合信号
        return t, samples
    
    def get_HFM_period(self, signal_type='1'):
        '''
        获取HFM信号周期
        :return:
        '''
        fs = self.signalfsSpinBox.value()  #采样频率
        T = self.signalTSpinBox.value()  #单周期长度
        NP = self.signalNPSpinBox.value()  #周期数
        t = np.arange(0, T, 1/fs)  #时间序列
        samples_zero = np.zeros_like(t)  #信号序列

        if signal_type == '1':
            f0 = self.sig1f0.value()  #起始频率
            f1 = self.sig1f1.value()  #终止频率
            t0 = self.sig1t0.value()  #起始时间
            t1 = self.sig1t1.value()  #终止时间
        elif signal_type == '2':
            f0 = self.sig2f0.value()
            f1 = self.sig2f1.value()
            t0 = self.sig2t0.value()
            t1 = self.sig2t1.value()
        elif signal_type == '3':
            f0 = self.sig3f0.value()
            f1 = self.sig3f1.value()
            t0 = self.sig3t0.value()
            t1 = self.sig3t1.value()

        samples_hfm = np.zeros_like(t)  #信号序列初始化
        pulse_mask = (t>=t0) & (t<=t1)  #脉冲信号的时间范围
        mu = (f1 - f0)/((t1-t0)*f0*f1)  #HFM信号的周期斜率
        t_adjusted = t[pulse_mask] - t0  #调整时间轴，使其相对于 t0!!!
        samples_hfm[pulse_mask] = np.cos(2*np.pi / mu * np.log(-mu * t_adjusted + 1/f0))  #脉冲信号
        samples = samples_zero + samples_hfm  #混合信号
        return t, samples

    def add_noise(self):
        '''
        叠加.wav文件噪声，并根据信噪比（SNR）调整噪声幅度
        :return:
            original_signal: 叠加噪声前的信号
            noise_signal: 噪声信号
            mixed_signal: 叠加噪声后的信号
        '''
        # 读取.wav文件
        wav_filepath = self.wavfilepath.text()
        if wav_filepath == '未选择wav文件':
            return None, None, None  # 如果没有选择.wav文件，直接返回

        # 读取.wav文件的采样频率和数据
        wav_fs, wav_data = wav.read(wav_filepath)

        # 如果.wav文件是立体声，只取其中一个声道
        if len(wav_data.shape) > 1:
            wav_data = wav_data[:, 0]  # 取左声道

        # 获取仿真信号的参数
        fs = self.signalfsSpinBox.value()  # 仿真信号的采样频率
        T = self.signalTSpinBox.value()  # 仿真信号的单周期长度
        NP = self.signalNPSpinBox.value()  # 获取周期数
        start_time = self.wavstarttimeSpinBox.value()  # 噪声的起始时间
        snr_db = self.snrSpinBox.value()  # 信噪比（dB）

        # 对噪声信号进行重采样，使其采样频率与仿真信号一致
        if wav_fs != fs:
            # 计算重采样后的长度
            resample_length = int(len(wav_data) * fs / wav_fs)
            wav_data = resample(wav_data, resample_length)  # 重采样
            wav_fs = fs  # 更新采样频率

        # 计算需要截取的噪声长度
        noise_duration = T * NP  # 噪声的持续时间为仿真信号的单周期长度
        noise_samples = int(noise_duration * fs)  # 噪声的样本数

        # 计算噪声在.wav文件中的起始样本点
        wav_start_sample = int(start_time * wav_fs)

        # 截取噪声信号
        if wav_start_sample + noise_samples > len(wav_data):
            print("警告：.wav文件长度不足，无法截取完整噪声信号！")
            return None, None, None
        noise_signal = wav_data[wav_start_sample:wav_start_sample + noise_samples]

        # 如果噪声信号长度不足，补零
        if len(noise_signal) < noise_samples:
            padding = np.zeros(noise_samples - len(noise_signal))
            noise_signal = np.concatenate((noise_signal, padding))

        # 计算信号功率和噪声功率
        signal_power = np.mean(self.mix_samples ** 2)  # 信号功率
        noise_power = np.mean(noise_signal ** 2)  # 噪声功率

        # 如果噪声功率为0（静音），直接返回
        if noise_power == 0:
            print("警告：噪声信号功率为0，无法叠加噪声！")
            return None, None, None

        # 根据信噪比（SNR）调整噪声幅度
        target_snr = 10 ** (snr_db / 10)  # 将 SNR 从 dB 转换为线性值
        scale_factor = np.sqrt(signal_power / (target_snr * noise_power))  # 计算缩放因子
        noise_signal = noise_signal * scale_factor  # 调整噪声信号幅度

        # 保存叠加噪声前的信号
        original_signal = self.mix_samples.copy()

        # 将噪声信号与仿真信号叠加
        mixed_signal = self.mix_samples + noise_signal[:len(self.mix_samples)]  # 确保噪声信号长度不超过仿真信号

        # 更新标志位
        self.is_add_noise = True

        # 返回叠加噪声前的信号、噪声信号和叠加噪声后的信号
        return original_signal, noise_signal, mixed_signal
    
    def btn_save_clicked(self):
        '''
        点击save按钮，保存画布上的图片至指定文件夹
        :return:
        '''
        # 打开文件对话框，选择保存路径
        filepath, _ = QFileDialog.getSaveFileName(
            self, "保存图片", "", "PNG Files (*.png);;JPEG Files (*.jpg);;All Files (*)"
        )

        # 如果用户选择了保存路径
        if filepath:
            try:
                # 保存当前画布上的图像
                self.stft_canvas.figure.savefig(filepath, dpi=300, bbox_inches='tight')
                print(f"图片已保存至：{filepath}")
            except Exception as e:
                print(f"保存图片失败：{e}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    mywindow = MainWindow()
    mywindow.show()
    sys.exit(app.exec_())
