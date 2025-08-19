import sys
import numpy as np
import nibabel as nib
from vtkmodules.vtkRenderingVolume import vtkGPUVolumeRayCastMapper

from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QFileDialog, QWidget, QVBoxLayout
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

import vtkmodules.all as vtk
from vtkmodules.vtkRenderingVolume import vtkGPUVolumeRayCastMapper

from vtkmodules.vtkCommonDataModel import vtkImageData
from vtkmodules.vtkCommonCore import vtkFloatArray
from vtkmodules.vtkCommonDataModel import vtkPiecewiseFunction
from vtkmodules.vtkRenderingCore import (
    vtkVolumeProperty,
    vtkVolume,
    vtkRenderer,
    vtkColorTransferFunction
)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("NIfTI 3D Viewer")
        self.resize(800, 600)

        # 主界面组件
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.layout = QVBoxLayout(self.central_widget)

        # 打开按钮
        self.open_button = QPushButton("打开 NIfTI 文件")
        self.layout.addWidget(self.open_button)

        # VTK 渲染窗口
        self.vtk_widget = QVTKRenderWindowInteractor(self.central_widget)
        self.layout.addWidget(self.vtk_widget)

        # 按钮绑定
        self.open_button.clicked.connect(self.open_file)

        # 初始化 VTK 相关
        self.renderer = vtkRenderer()
        self.vtk_widget.GetRenderWindow().AddRenderer(self.renderer)
        self.vtk_widget.Initialize()

    def open_file(self):
        # 打开文件对话框
        file_name, _ = QFileDialog.getOpenFileName(self, "选择 NIfTI 文件", "", "NIfTI Files (*.nii *.nii.gz)")
        if file_name:
            self.display_nifti(file_name)

    def display_nifti(self, nifti_path):
        # 清除原先渲染内容
        self.renderer.RemoveAllViewProps()

        # 加载 NIfTI 文件
        nii = nib.load(nifti_path)
        data = nii.get_fdata()

        # 创建 VTK 图像数据
        img_data = vtkImageData()
        img_data.SetDimensions(data.shape[0], data.shape[1], data.shape[2])
        img_data.SetSpacing(nii.header.get_zooms())

        vtk_array = vtkFloatArray()
        vtk_array.SetNumberOfValues(np.prod(data.shape))
        vtk_array.SetArray(data.astype(np.float32).ravel(), np.prod(data.shape), 1)
        img_data.GetPointData().SetScalars(vtk_array)

        # 设置 Volume Mapper
        volume_mapper = vtkGPUVolumeRayCastMapper()
        volume_mapper.SetInputData(img_data)

        # 设置透明度
        opacity = vtkPiecewiseFunction()
        opacity.AddPoint(0, 0.0)
        opacity.AddPoint(50, 0.05)
        opacity.AddPoint(1000, 0.2)
        opacity.AddPoint(3000, 0.6)

        # 设置颜色
        color = vtkColorTransferFunction()
        color.AddRGBPoint(0, 0.0, 0.0, 0.0)
        color.AddRGBPoint(1000, 1.0, 1.0, 1.0)

        # Volume 属性
        volume_property = vtkVolumeProperty()
        volume_property.SetColor(color)
        volume_property.SetScalarOpacity(opacity)
        volume_property.ShadeOff()
        volume_property.SetInterpolationTypeToLinear()

        # Volume
        volume = vtkVolume()
        volume.SetMapper(volume_mapper)
        volume.SetProperty(volume_property)

        # 加到渲染器
        self.renderer.AddVolume(volume)
        self.renderer.ResetCamera()

        # 渲染刷新
        self.vtk_widget.GetRenderWindow().Render()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


