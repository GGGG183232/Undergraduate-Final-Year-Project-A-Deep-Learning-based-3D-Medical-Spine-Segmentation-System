import sys
import os
import torch
import numpy as np
import nibabel as nib
import yaml
import vtk
import nibabel as nib
import numpy as np
from vtkmodules.vtkRenderingCore import vtkRenderer, vtkRenderWindow, vtkRenderWindowInteractor, vtkVolume, vtkVolumeProperty, vtkColorTransferFunction
from vtkmodules.vtkRenderingVolume import vtkGPUVolumeRayCastMapper
from vtkmodules.vtkCommonDataModel import vtkPiecewiseFunction
from PyQt5.QtGui import QPixmap, QImage

from vtkmodules.vtkCommonDataModel import vtkImageData
from vtkmodules.vtkCommonCore import vtkFloatArray


from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout,
    QFileDialog, QTextEdit, QProgressBar, QHBoxLayout
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QThread, pyqtSignal

from model.vnet import VNet
from evaluation.dice_score import dice_score
import matplotlib.pyplot as plt

class InferenceThread(QThread):
    finished = pyqtSignal(float, str)  # 传出dice系数和推理结果路径
    progress = pyqtSignal(int)

    def __init__(self, model, device, input_path, output_path):
        super().__init__()
        self.model = model
        self.device = device
        self.input_path = input_path
        self.output_path = output_path

    def run(self):
        self.progress.emit(10)

        nii = nib.load(self.input_path)
        volume = nii.get_fdata()
        volume = np.expand_dims(volume, axis=0)  # [1, D, H, W]
        volume = np.expand_dims(volume, axis=0)  # [1, 1, D, H, W]

        volume = torch.tensor(volume, dtype=torch.float32).to(self.device)
        self.model.eval()
        with torch.no_grad():
            output = self.model(volume)
            output = (torch.sigmoid(output) > 0.5).float()

        self.progress.emit(70)

        output_np = output.cpu().numpy()[0, 0]  # [D, H, W]

        # 保存预测为nii
        output_nii = nib.Nifti1Image(output_np, affine=nii.affine)
        nib.save(output_nii, self.output_path)

        # 计算dice
        mask = nii.get_fdata()
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
        dice = dice_score(output, mask)

        self.progress.emit(100)
        self.finished.emit(dice.item(), self.output_path)


class PredictApp(QWidget):
    def __init__(self):
        super().__init__()

        self.initUI()
        self.model, self.device = self.load_model()
        self.input_path = None
        self.output_path = "./predicted_output.nii.gz"

    def initUI(self):
        self.setWindowTitle('VNet CT Segmentation')
        self.setGeometry(100, 100, 800, 600)

        layout = QVBoxLayout()

        # 按钮
        button_layout = QHBoxLayout()
        self.select_button = QPushButton('选择CT NII文件')
        self.select_button.clicked.connect(self.select_file)
        button_layout.addWidget(self.select_button)

        self.predict_button = QPushButton('开始预测')
        self.predict_button.clicked.connect(self.start_inference)
        button_layout.addWidget(self.predict_button)

        layout.addLayout(button_layout)

        # 图像显示
        img_layout = QHBoxLayout()
        self.input_label = QLabel('原图')
        self.input_label.setFixedSize(360, 360)
        self.input_label.setStyleSheet("border: 1px solid black;")
        self.input_label.setAlignment(Qt.AlignCenter)
        img_layout.addWidget(self.input_label)

        self.output_label = QLabel('预测结果')
        self.output_label.setFixedSize(360, 360)
        self.output_label.setStyleSheet("border: 1px solid black;")
        self.output_label.setAlignment(Qt.AlignCenter)
        img_layout.addWidget(self.output_label)

        layout.addLayout(img_layout)

        # 进度条
        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)

        # 正确率输出
        self.text_edit = QTextEdit()
        self.text_edit.setReadOnly(True)
        layout.addWidget(self.text_edit)

        self.setLayout(layout)

    def load_model(self):
        # 读取config
        with open("./config.yaml", "r") as infile:
            config = yaml.load(infile, Loader=yaml.FullLoader)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = VNet()
        model.to(device)

        weight_path = 'result/bestmodel/9.11_128128128/Epoch149_LS-0.951_DC-0.826.pth'
        checkpoint = torch.load(weight_path, map_location=device)
        model.load_state_dict(checkpoint)

        return model, device

    def select_file(self):
        file_name, _ = QFileDialog.getOpenFileName(self, '选择NIfTI文件', '', 'NIfTI Files (*.nii *.nii.gz)')
        if file_name:
            self.input_path = file_name
            self.display_nifti_slice(file_name, self.input_label)

    def start_inference(self):
        if not self.input_path:
            self.text_edit.append("请先选择一张NIfTI文件！")
            return

        self.thread = InferenceThread(self.model, self.device, self.input_path, self.output_path)
        self.thread.progress.connect(self.update_progress)
        self.thread.finished.connect(self.inference_done)
        self.thread.start()

    def update_progress(self, val):
        self.progress_bar.setValue(val)

    def inference_done(self, dice, output_path):
        self.text_edit.append(f"推理完成！Dice系数: {dice:.4f}")
        self.display_nifti_slice(output_path, self.output_label)

    # def display_nifti_slice(self, nifti_path, label_widget):
    #     nii = nib.load(nifti_path)
    #     data = nii.get_fdata()
    #     mid_slice = data.shape[2] // 2  # 中间一张
    #     img = data[:, :, mid_slice]
    #
    #     img = np.clip(img, 0, np.percentile(img, 99))  # 防止极值影响显示
    #     img = (img - np.min(img)) / (np.max(img) - np.min(img)) * 255.0
    #     img = img.astype(np.uint8)
    #
    #     h, w = img.shape
    #     img_bytes = img.tobytes()  # 转换为bytes类型
    #     qimg = QImage(img_bytes, w, h, w, QImage.Format_Grayscale8)
    #     pixmap = QPixmap.fromImage(qimg).scaled(label_widget.width(), label_widget.height(), Qt.KeepAspectRatio)
    #     label_widget.setPixmap(pixmap)

    from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
    from vtkmodules.vtkRenderingVolume import vtkGPUVolumeRayCastMapper
    from vtkmodules.vtkRenderingCore import vtkVolume, vtkVolumeProperty, vtkRenderer, vtkRenderWindow
    from vtkmodules.vtkCommonDataModel import vtkImageData
    from vtkmodules.vtkCommonCore import vtkFloatArray
    from vtkmodules.vtkCommonDataModel import vtkPiecewiseFunction
    from vtkmodules.vtkRenderingCore import vtkColorTransferFunction
    import nibabel as nib
    import numpy as np
    import vtk

    from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
    from vtkmodules.vtkRenderingVolume import vtkGPUVolumeRayCastMapper
    from vtkmodules.vtkRenderingCore import vtkVolume, vtkVolumeProperty, vtkRenderer, vtkRenderWindow
    from vtkmodules.vtkCommonDataModel import vtkImageData
    from vtkmodules.vtkCommonCore import vtkFloatArray
    from vtkmodules.vtkCommonDataModel import vtkPiecewiseFunction
    from vtkmodules.vtkRenderingCore import vtkColorTransferFunction
    import nibabel as nib
    import numpy as np
    import vtk

    def display_nifti_slice(self, nifti_path, vtk_widget: QVTKRenderWindowInteractor):
        # 1. 加载 NIfTI 文件
        nii = nib.load(nifti_path)
        data = nii.get_fdata()

        # 2. 创建 VTK 图像
        img_data = vtkImageData()
        img_data.SetDimensions(data.shape)
        img_data.SetSpacing(nii.header.get_zooms())

        # 设置数据
        vtk_array = vtkFloatArray()
        vtk_array.SetName("ImageData")
        vtk_array.SetNumberOfValues(np.prod(data.shape))
        vtk_array.SetArray(data.astype(np.float32).ravel(), np.prod(data.shape), 1)
        img_data.GetPointData().SetScalars(vtk_array)

        # 3. 设置 Volume 渲染
        volume_mapper = vtkGPUVolumeRayCastMapper()
        volume_mapper.SetInputData(img_data)

        opacity = vtkPiecewiseFunction()
        opacity.AddPoint(0, 0.0)
        opacity.AddPoint(50, 0.05)
        opacity.AddPoint(1000, 0.2)
        opacity.AddPoint(3000, 0.6)

        color = vtkColorTransferFunction()
        color.AddRGBPoint(0, 0.0, 0.0, 0.0)
        color.AddRGBPoint(1000, 1.0, 1.0, 1.0)

        volume_property = vtkVolumeProperty()
        volume_property.SetColor(color)
        volume_property.SetScalarOpacity(opacity)
        volume_property.ShadeOff()
        volume_property.SetInterpolationTypeToLinear()

        volume = vtkVolume()
        volume.SetMapper(volume_mapper)
        volume.SetProperty(volume_property)

        # 4. 设置嵌入式 Renderer 和窗口
        renderer = vtkRenderer()
        renderer.AddVolume(volume)
        renderer.SetBackground(0.0, 0.0, 0.0)

        render_window = vtk_widget.GetRenderWindow()
        render_window.AddRenderer(renderer)

        interactor = vtk_widget
        interactor.Initialize()
        interactor.Start()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = PredictApp()
    ex.show()
    sys.exit(app.exec_())
