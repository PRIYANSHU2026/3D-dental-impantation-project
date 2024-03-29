import os
import pydicom
import vtk
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QFileDialog, QVBoxLayout, QHBoxLayout, QSlider, QCheckBox
from PyQt5.QtCore import Qt
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from segmentation_models import Unet
from keras.models import load_model

class DicomRenderer(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.choose_directory_button = QPushButton('Choose DICOM Directory', self)
        self.choose_directory_button.clicked.connect(self.chooseDirectory)

        self.vtk_renderer = vtk.vtkRenderer()
        self.vtk_render_window = vtk.vtkRenderWindow()
        self.vtk_render_window.SetWindowName("Dental 3D Rendering")
        self.vtk_render_window.SetSize(400, 400)
        self.vtk_render_window.AddRenderer(self.vtk_renderer)

        # Use QVTKRenderWindowInteractor for integration with Qt
        self.vtk_render_window_interactor = QVTKRenderWindowInteractor(self)
        self.vtk_render_window_interactor.SetRenderWindow(self.vtk_render_window)

        # Matplotlib figure and canvas
        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)

        # Slider for navigating through slices
        self.slice_slider = QSlider(Qt.Horizontal)
        self.slice_slider.valueChanged.connect(self.updateSlice)

        # Checkbox for mesh cut-out mode
        self.cutout_checkbox = QCheckBox("Mesh Cut-Out Mode")
        self.cutout_checkbox.stateChanged.connect(self.toggleCutoutMode)

        # Checkbox for marking mode
        self.marking_checkbox = QCheckBox("Marking Mode")
        self.marking_checkbox.stateChanged.connect(self.toggleMarkingMode)

        layout = QHBoxLayout(self)

        # Left side for 3D rendering
        vtk_container = QWidget(self)
        vtk_layout = QVBoxLayout(vtk_container)
        vtk_layout.addWidget(self.choose_directory_button)
        vtk_layout.addWidget(self.vtk_render_window_interactor)
        vtk_layout.addWidget(self.slice_slider)
        vtk_layout.addWidget(self.cutout_checkbox)
        vtk_layout.addWidget(self.marking_checkbox)
        layout.addWidget(vtk_container)

        # Right side for Matplotlib with slider
        matplotlib_container = QWidget(self)
        matplotlib_layout = QVBoxLayout(matplotlib_container)
        matplotlib_layout.addWidget(self.toolbar)
        matplotlib_layout.addWidget(self.canvas)
        layout.addWidget(matplotlib_container)

        self.setGeometry(300, 300, 800, 400)
        self.setWindowTitle('DICOM Renderer')

        self.dicom_files = []
        self.current_slice = 0
        self.directory_path = ""
        self.marking_points = []
        self.marking_mode_enabled = False
        self.yellow_markers_3d = []

        # Initialize the segmentation model
        self.tooth_segmentation_model = self.initializeSegmentationModel()

    def initializeSegmentationModel(self):
        # Load the segmentation model
        model = Unet('resnet34', classes=1, activation='sigmoid')

        # Load the pretrained weights
        weights_path = '/Users/shikarichacha/Downloads/resnet34_imagenet_1000_no_top.h5'  # Update with your actual path

        # Check if the weights file exists
        if os.path.exists(weights_path):
            # Load the weights using the appropriate method
            try:
                model.load_weights(weights_path)
            except Exception as e:
                print(f"Error loading weights: {e}")
        else:
            print(f"Weights file not found at: {weights_path}")

        return model

    def loadDicomAndRender(self, directory_path):
        self.dicom_files = [f for f in os.listdir(directory_path) if f.endswith(".dcm")]
        self.dicom_files.sort(key=lambda x: int(x.split('Slice')[1].split('.dcm')[0]))

        first_dcm_file = pydicom.read_file(os.path.join(directory_path, self.dicom_files[0]))
        rows, cols = first_dcm_file.Rows, first_dcm_file.Columns

        volume_data = np.zeros((len(self.dicom_files), rows, cols), dtype=np.uint16)

        for i, filename in enumerate(self.dicom_files):
            file_path = os.path.join(directory_path, filename)
            dcm_file = pydicom.read_file(file_path)
            volume_data[i, :, :] = dcm_file.pixel_array

        vtk_volume = vtk.vtkImageData()
        vtk_volume.SetDimensions(cols, rows, len(self.dicom_files))
        vtk_volume.AllocateScalars(vtk.VTK_UNSIGNED_SHORT, 1)

        vtk_np_array_flat = volume_data.ravel()
        vtk_data_array = vtk.vtkUnsignedShortArray()
        vtk_data_array.SetArray(vtk_np_array_flat, len(vtk_np_array_flat), 1)
        vtk_volume.GetPointData().SetScalars(vtk_data_array)

        marching_cubes = vtk.vtkMarchingCubes()
        marching_cubes.SetInputData(vtk_volume)
        marching_cubes.SetValue(0, 1500)

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(marching_cubes.GetOutputPort())

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)

        self.vtk_renderer.AddActor(actor)
        self.vtk_renderer.ResetCamera()

        # Color the missing teeth in yellow
        missing_teeth = [1, 3]  # Replace with the actual list of missing teeth indices

        color_array = vtk.vtkUnsignedCharArray()
        color_array.SetNumberOfComponents(3)
        color_array.SetName("Colors")

        for i in range(mapper.GetInput().GetNumberOfPoints()):
            color = [255, 255, 255]  # Default color (white)

            for tooth_idx in missing_teeth:
                tooth_location_x = 50  # Replace with the actual X-coordinate
                tooth_location_y = 50  # Replace with the actual Y-coordinate

                # Define a region around the missing tooth to color in yellow
                region_x_min, region_x_max = tooth_location_x - 5, tooth_location_x + 5
                region_y_min, region_y_max = tooth_location_y - 5, tooth_location_y + 5

                # Check if the current point is in the region of the missing tooth
                point = mapper.GetInput().GetPoint(i)
                if region_x_min <= point[0] <= region_x_max and region_y_min <= point[1] <= region_y_max:
                    color = [255, 255, 0]  # Yellow color for the missing tooth
                    break  # Break out of the loop if the point is in the region

            color_array.InsertNextTuple(color)

        mapper.GetInput().GetPointData().SetScalars(color_array)

        self.vtk_render_window.Render()

        # Display the current DICOM slice using Matplotlib
        self.displayDicomSlice()

    def displayDicomSlice(self):
        self.figure.clear()
        self.ax = self.figure.add_subplot(111)

        filename = self.dicom_files[self.current_slice]
        dicom_path = os.path.join(self.directory_path, filename)
        ds = pydicom.read_file(dicom_path)
        pixel_array = ds.pixel_array
        self.ax.imshow(pixel_array, cmap='gray', aspect='auto')
        self.ax.set_title(f'DICOM Slice {self.current_slice + 1}/{len(self.dicom_files)}')
        self.ax.set_axis_off()

        # Highlight missing teeth in red in 2D view
        missing_teeth = [1, 3]  # Replace with the actual list of missing teeth indices
        for tooth_idx in missing_teeth:
            tooth_location_x = 50  # Replace with the actual X-coordinate
            tooth_location_y = 50  # Replace with the actual Y-coordinate
            self.ax.scatter(tooth_location_x, tooth_location_y, s=100, c='red', marker='X', label=f"Missing {tooth_idx}")

        # Marking points on the current slice
        for point in self.marking_points:
            self.ax.plot(point[0], point[1], 'ro')  # Assuming marking points are (x, y) coordinates

        # Yellow markers in 3D
        for marker_position in self.yellow_markers_3d:
            marker = vtk.vtkSphereSource()
            marker.SetCenter(marker_position[0], marker_position[1], marker_position[2])
            marker.SetRadius(5.0)
            marker_mapper = vtk.vtkPolyDataMapper()
            marker_mapper.SetInputConnection(marker.GetOutputPort())
            marker_actor = vtk.vtkActor()
            marker_actor.SetMapper(marker_mapper)
            marker_actor.GetProperty().SetColor(1.0, 1.0, 0.0)  # Yellow color
            self.vtk_renderer.AddActor(marker_actor)

        self.ax.legend()  # Display the legend for the missing teeth markers
        self.canvas.draw()

    def chooseDirectory(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ShowDirsOnly | QFileDialog.DontUseNativeDialog
        self.directory_path = QFileDialog.getExistingDirectory(self, 'Select DICOM Directory', options=options)

        if self.directory_path:
            self.loadDicomAndRender(self.directory_path)
            self.slice_slider.setRange(0, len(self.dicom_files) - 1)
            self.slice_slider.setValue(0)

    def toggleCutoutMode(self, state):
        actor_collection = self.vtk_renderer.GetActors()

        if actor_collection.GetNumberOfItems() > 0:
            actor = actor_collection.GetItemAsObject(0)

            if state == Qt.Checked:
                # Enable mesh cut-out mode
                actor.GetProperty().SetOpacity(0.5)
            else:
                # Disable mesh cut-out mode
                actor.GetProperty().SetOpacity(1.0)

            self.vtk_render_window.Render()

    def toggleMarkingMode(self, state):
        self.marking_mode_enabled = state == Qt.Checked

    def updateSlice(self):
        self.current_slice = self.slice_slider.value()
        self.displayDicomSlice()

        # Update mesh visibility based on cut-out mode
        self.toggleCutoutMode(self.cutout_checkbox.isChecked())

        # Update marking points on the current slice
        if self.marking_mode_enabled:
            self.marking_points = self.getMarkingPointsFromUser()

    def getMarkingPointsFromUser(self):
        self.marking_points = []

        def on_click(event):
            if event.inaxes is not None:
                x, y = event.xdata, event.ydata
                self.marking_points.append((x, y))
                self.displayDicomSlice()  # Update display after each marking

        # Connect the on_click function to mouse click events
        self.canvas.mpl_connect('button_press_event', on_click)

        # Provide instructions to the user
        print("Marking Mode: Click on points to mark. Press 'q' to exit marking mode.")

        # Wait for the user to finish marking (press 'q' to exit)
        plt.show(block=True)

        # Disconnect the event handler to prevent further interaction
        self.canvas.mpl_disconnect('button_press_event')

        return self.marking_points

    def addYellowMarker3D(self, x, y, z):
        self.yellow_markers_3d.append((x, y, z))
        self.displayDicomSlice()

if __name__ == '__main__':
    app = QApplication([])
    dicom_renderer = DicomRenderer()
    dicom_renderer.show()
    app.exec_()
