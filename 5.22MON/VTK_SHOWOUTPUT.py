import itk
from vtkmodules.vtkCommonColor import vtkNamedColors
from vtkmodules.vtkFiltersGeneral import vtkDiscreteMarchingCubes
from vtkmodules.vtkRenderingCore import vtkActor, vtkPolyDataMapper, vtkRenderer, \
    vtkRenderWindow, vtkRenderWindowInteractor

def show_3d_nifti_image(nifti_file_name):

    # Read NIFTI file
    itk_img = itk.imread(filename=nifti_file_name)

    # Convert itk to vtk
    vtk_img = itk.vtk_image_from_image(l_image=itk_img)

    # Extract vtkImageData contour to vtkPolyData
    contour = vtkDiscreteMarchingCubes()
    contour.SetInputData(vtk_img)

    # Define colors, mapper, actor, renderer, renderWindow, renderWindowInteractor
    colors = vtkNamedColors()

    mapper = vtkPolyDataMapper()
    mapper.SetInputConnection(contour.GetOutputPort())

    actor = vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(colors.GetColor3d("White"))  # 设置物体为白色！

    renderer = vtkRenderer()
    renderer.AddActor(actor)
    renderer.SetBackground(colors.GetColor3d("Black"))  # 设置背景为黑色！

    renderWindow = vtkRenderWindow()
    renderWindow.AddRenderer(renderer)
    renderWindow.SetSize(600, 600)  # 设置窗口大小

    renderWindowInteractor = vtkRenderWindowInteractor()
    renderWindowInteractor.SetRenderWindow(renderWindow)
    renderWindow.Render()
    renderWindowInteractor.Start()

if __name__ == '__main__':
    #show_3d_nifti_image("data/dataset-verse20test/rawdata/sub-gl108/sub-gl108_dir-ax_ct.nii.gz")
    show_3d_nifti_image("data/dataset-verse20test/derivatives/sub-gl108/sub-gl108_dir-ax_seg-vert_msk.nii.gz")
