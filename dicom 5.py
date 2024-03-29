import os
import pydicom
import vtk
import numpy as np

def load_dicom_series(folder_path):
    dicom_files = [f for f in os.listdir(folder_path) if f.endswith(".dcm")]
    dicom_files.sort()

    first_dcm_file = pydicom.read_file(os.path.join(folder_path, dicom_files[0]))
    rows, cols = first_dcm_file.Rows, first_dcm_file.Columns

    volume_data = np.zeros((len(dicom_files), rows, cols), dtype=np.int16)

    for i, filename in enumerate(dicom_files):
        file_path = os.path.join(folder_path, filename)
        dcm_file = pydicom.read_file(file_path)
        volume_data[i, :, :] = dcm_file.pixel_array

    vtk_volume = vtk.vtkImageData()
    vtk_volume.SetDimensions(cols, rows, len(dicom_files))
    vtk_volume.AllocateScalars(vtk.VTK_SHORT, 1)

    vtk_np_array_flat = volume_data.ravel()
    vtk_data_array = vtk.vtkShortArray()
    vtk_data_array.SetArray(vtk_np_array_flat, len(vtk_np_array_flat), 1)
    vtk_volume.GetPointData().SetScalars(vtk_data_array)

    return vtk_volume

def create_marching_cubes(vtk_volume, threshold_value):
    marching_cubes = vtk.vtkMarchingCubes()
    marching_cubes.SetInputData(vtk_volume)
    marching_cubes.SetValue(0, threshold_value)
    return marching_cubes

def create_renderer(mapper):
    renderer = vtk.vtkRenderer()
    renderer.SetBackground(1, 1, 1)  # Set background color to white

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)

    renderer.AddActor(actor)
    renderer.ResetCamera()

    return renderer

def main():
    dicom_folder = "/Users/shikarichacha/Downloads/3d segmentation"
    threshold_value = 300

    vtk_volume = load_dicom_series(dicom_folder)
    marching_cubes = create_marching_cubes(vtk_volume, threshold_value)

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(marching_cubes.GetOutputPort())

    renderer = create_renderer(mapper)

    render_window = vtk.vtkRenderWindow()
    render_window.SetWindowName("3D Visualization")
    render_window.SetSize(800, 800)
    render_window.AddRenderer(renderer)

    render_window_interactor = vtk.vtkRenderWindowInteractor()
    render_window_interactor.SetRenderWindow(render_window)

    render_window.Render()
    render_window_interactor.Start()

if __name__ == "__main__":
    main()
