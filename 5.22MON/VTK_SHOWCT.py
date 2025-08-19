import sys
import itk
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QFileDialog, QVBoxLayout, QWidget

from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from vtkmodules.vtkRenderingCore import vtkRenderer, vtkRenderWindowInteractor, vtkVolume, vtkVolumeProperty, vtkColorTransferFunction
from vtkmodules.vtkRenderingVolume import vtkGPUVolumeRayCastMapper
from vtkmodules.vtkCommonDataModel import vtkPiecewiseFunction


class NiftiVolumeViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("NIfTI 体渲染查看器")
        self.setGeometry(100, 100, 800, 800)

        # 按钮：选择文件
        self.button = QPushButton("选择 NIfTI 图像")
        self.button.clicked.connect(self.open_file)

        # VTK 渲染控件
        self.vtk_widget = QVTKRenderWindowInteractor()
        self.renderer = vtkRenderer()
        self.vtk_widget.GetRenderWindow().AddRenderer(self.renderer)

        # 布局
        layout = QVBoxLayout()
        layout.addWidget(self.button)
        layout.addWidget(self.vtk_widget)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        # 初始化交互器
        self.vtk_widget.Initialize()
        self.vtk_widget.Start()

    def open_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择 NIfTI 图像", "", "NIfTI Files (*.nii *.nii.gz)")
        if file_path:
            self.render_nifti(file_path)

    def render_nifti(self, nifti_file_name):
        # 清空原先的渲染
        self.renderer.RemoveAllViewProps()

        # Read NIFTI file using ITK
        itk_img = itk.imread(nifti_file_name)

        # Convert ITK image to VTK image
        vtk_img = itk.vtk_image_from_image(itk_img)

        # Set up volume rendering mapper
        volume_mapper = vtkGPUVolumeRayCastMapper()
        volume_mapper.SetInputData(vtk_img)

        # Set up opacity transfer function
        opacity = vtkPiecewiseFunction()
        opacity.AddPoint(0, 0.0)
        opacity.AddPoint(50, 0.05)
        opacity.AddPoint(1000, 0.2)
        opacity.AddPoint(3000, 0.6)

        # Set up color transfer function (grayscale mapping)
        color = vtkColorTransferFunction()
        color.AddRGBPoint(0, 0.0, 0.0, 0.0)
        color.AddRGBPoint(1000, 1.0, 1.0, 1.0)

        # Set volume property
        volume_property = vtkVolumeProperty()
        volume_property.SetColor(color)
        volume_property.SetScalarOpacity(opacity)
        volume_property.ShadeOff()
        volume_property.SetInterpolationTypeToLinear()

        # Create volume actor
        volume = vtkVolume()
        volume.SetMapper(volume_mapper)
        volume.SetProperty(volume_property)

        # 渲染
        self.renderer.AddVolume(volume)
        self.renderer.SetBackground(0.0, 0.0, 0.0)
        self.renderer.ResetCamera()
        self.vtk_widget.GetRenderWindow().Render()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = NiftiVolumeViewer()
    window.show()
    sys.exit(app.exec_())
