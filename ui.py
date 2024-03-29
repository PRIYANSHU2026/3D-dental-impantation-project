import os
import pydicom
import vtk
import numpy as np
from tkinter import Tk, filedialog, Button

def load_dicom_and_render(directory_path):
    # Collect all DICOM files in the directory
    dicom_files = [f for f in os.listdir(directory_path) if f.endswith(".dcm")]

    # Sort the files based on their numerical order (assuming filenames contain numbers)
    dicom_files.sort(key=lambda x: int(x.split('Slice')[1].split('.dcm')[0]))

    # Read the first DICOM file to get the dimensions
    first_dcm_file = pydicom.read_file(os.path.join(directory_path, dicom_files[0]))
    rows, cols = first_dcm_file.Rows, first_dcm_file.Columns

    # Create a 3D array to store the pixel data from all DICOM files
    volume_data = np.zeros((len(dicom_files), rows, cols), dtype=np.uint16)

    # Iterate over all DICOM files
    for i, filename in enumerate(dicom_files):
        file_path = os.path.join(directory_path, filename)

        # Read DICOM file
        dcm_file = pydicom.read_file(file_path)
        volume_data[i, :, :] = dcm_file.pixel_array

    # Create a VTK volume
    vtk_volume = vtk.vtkImageData()
    vtk_volume.SetDimensions(cols, rows, len(dicom_files))
    vtk_volume.AllocateScalars(vtk.VTK_UNSIGNED_SHORT, 1)

    # Flatten and copy the pixel data to VTK volume
    vtk_np_array_flat = volume_data.ravel()
    vtk_data_array = vtk.vtkUnsignedShortArray()
    vtk_data_array.SetArray(vtk_np_array_flat, len(vtk_np_array_flat), 1)
    vtk_volume.GetPointData().SetScalars(vtk_data_array)

    # Create a vtkMarchingCubes to extract tooth structures
    marching_cubes = vtk.vtkMarchingCubes()
    marching_cubes.SetInputData(vtk_volume)
    marching_cubes.SetValue(0, 1500)  # Adjust this threshold value based on your DICOM data

    # Create a vtkPolyDataMapper
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(marching_cubes.GetOutputPort())

    # Create a vtkActor
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)

    # Create a vtkRenderer
    renderer = vtk.vtkRenderer()
    renderer.SetBackground(1, 1, 1)  # Set background color to white

    # Create a vtkRenderWindow
    render_window = vtk.vtkRenderWindow()
    render_window.SetWindowName("Dental 3D Rendering")
    render_window.SetSize(800, 800)
    render_window.AddRenderer(renderer)

    # Create a vtkRenderWindowInteractor
    render_window_interactor = vtk.vtkRenderWindowInteractor()
    render_window_interactor.SetRenderWindow(render_window)

    # Add the actor to the renderer
    renderer.AddActor(actor)

    # Set up a camera to view the entire volume
    renderer.ResetCamera()

    # Start the rendering loop
    render_window.Render()
    render_window_interactor.Start()

def choose_directory():
    root = Tk()
    root.withdraw()  # Hide the main window

    directory_path = filedialog.askdirectory(title="Select DICOM Directory")

    if directory_path:
        load_dicom_and_render(directory_path)

# Create a button to choose the directory
choose_directory_button = Button(
    text="Choose DICOM Directory",
    command=choose_directory
)
choose_directory_button.pack()

# Main Tkinter loop
Tk().mainloop()
