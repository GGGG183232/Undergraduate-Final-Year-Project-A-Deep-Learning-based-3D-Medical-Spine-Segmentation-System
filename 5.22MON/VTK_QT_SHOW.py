import sys
import itk
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QFileDialog

from vtkmodules.vtkCommonColor import vtkNamedColors
from vtkmodules.vtkFiltersGeneral import vtkDiscreteMarchingCubes
from vtkmodules.vtkRenderingCore import (
    vtkRenderer,
    vtkRenderWindow,
    vtkRenderWindowInteractor,
    vtkActor,
    vtkPolyDataMapper
)
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor


class Nifti3DViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("3D NIfTI Viewer (Discrete Marching Cubes)")
        self.setGeometry(100, 100, 800, 600)

        # 初始化 VTK 控件和按钮
        self.vtk_widget = QVTKRenderWindowInteractor(self)
        self.select_button = QPushButton("选择 NIfTI 图像")
        self.select_button.clicked.connect(self.select_nifti_file)

        # 布局设置
        layout = QVBoxLayout()
        layout.addWidget(self.select_button)
        layout.addWidget(self.vtk_widget)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        # 渲染器设置
        self.renderer = vtkRenderer()
        self.vtk_widget.GetRenderWindow().AddRenderer(self.renderer)
        self.iren = self.vtk_widget.GetRenderWindow().GetInteractor()
        self.iren.Initialize()

    def select_nifti_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择 NIfTI 图像", "", "NIfTI Files (*.nii *.nii.gz)"
        )
        if file_path:
            self.render_nifti(file_path)

    def render_nifti(self, file_path):
        # 清空原有渲染内容
        self.renderer.RemoveAllViewProps()

        # 读取 NIfTI 图像
        itk_img = itk.imread(file_path)
        vtk_img = itk.vtk_image_from_image(itk_img)

        # 离散Marching Cubes提取等值面（假设是label mask）
        contour = vtkDiscreteMarchingCubes()
        contour.SetInputData(vtk_img)
        contour.GenerateValues(1, 1, 1)  # 提取标签为1的区域

        # 设置 mapper 和 actor
        mapper = vtkPolyDataMapper()
        mapper.SetInputConnection(contour.GetOutputPort())

        actor = vtkActor()
        actor.SetMapper(mapper)

        colors = vtkNamedColors()
        self.renderer.AddActor(actor)
        self.renderer.SetBackground(colors.GetColor3d("SlateGray"))
        self.renderer.ResetCamera()

        self.vtk_widget.GetRenderWindow().Render()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    viewer = Nifti3DViewer()
    viewer.show()
    sys.exit(app.exec_())
