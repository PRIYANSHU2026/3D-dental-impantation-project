import os
import pydicom
import vtk
import numpy as np

directory_path = "/Users/shikarichacha/Downloads/3d segmentation"

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

# Create a vtkImageThreshold to identify potential fractures or missing teeth
threshold_filter = vtk.vtkImageThreshold()
threshold_filter.SetInputData(vtk_volume)
threshold_filter.ThresholdByLower(1500)  # Adjust this threshold value based on your DICOM data
threshold_filter.ReplaceInOn()
threshold_filter.SetInValue(0)
threshold_filter.ReplaceOutOn()
threshold_filter.SetOutValue(1)
threshold_filter.Update()

# Create a vtkMarchingCubes to extract the defective structures
defective_marching_cubes = vtk.vtkMarchingCubes()
defective_marching_cubes.SetInputConnection(threshold_filter.GetOutputPort())
defective_marching_cubes.SetValue(0, 1)

# Create a vtkPolyDataMapper for the defective structures
defective_mapper = vtk.vtkPolyDataMapper()
defective_mapper.SetInputConnection(defective_marching_cubes.GetOutputPort())

# Create a vtkActor for the defective structures and set its color to red
defective_actor = vtk.vtkActor()
defective_actor.SetMapper(defective_mapper)
defective_actor.GetProperty().SetColor(1, 0, 0)  # RGB color for red

# Add the defective actor to the renderer
renderer.AddActor(defective_actor)

# The rest of your VTK pipeline goes here...



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
