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
from skimage import measure
from segmentation_models import Unet
from keras.models import load_model
import cv2  # Make sure to install OpenCV: pip install opencv-python

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

        self.vtk_render_window_interactor = QVTKRenderWindowInteractor(self)
        self.vtk_render_window_interactor.SetRenderWindow(self.vtk_render_window)

        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)

        self.slice_slider = QSlider(Qt.Horizontal)
        self.slice_slider.valueChanged.connect(self.updateSlice)

        self.cutout_checkbox = QCheckBox("Mesh Cut-Out Mode")
        self.cutout_checkbox.stateChanged.connect(self.toggleCutoutMode)

        self.marking_checkbox = QCheckBox("Marking Mode")
        self.marking_checkbox.stateChanged.connect(self.toggleMarkingMode)

        layout = QHBoxLayout(self)

        vtk_container = QWidget(self)
        vtk_layout = QVBoxLayout(vtk_container)
        vtk_layout.addWidget(self.choose_directory_button)
        vtk_layout.addWidget(self.vtk_render_window_interactor)
        vtk_layout.addWidget(self.slice_slider)
        vtk_layout.addWidget(self.cutout_checkbox)
        vtk_layout.addWidget(self.marking_checkbox)
        layout.addWidget(vtk_container)

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

        self.tooth_segmentation_model = self.initializeSegmentationModel()

    def initializeSegmentationModel(self):
        model = Unet('resnet34', classes=1, activation='sigmoid')
        weights_path = '/Users/shikarichacha/Downloads/resnet34_imagenet_1000_no_top.h5'

        if os.path.exists(weights_path):
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

        missing_teeth = [1, 3]  # Replace with the actual list of missing teeth indices

        color_array = vtk.vtkUnsignedCharArray()
        color_array.SetNumberOfComponents(3)
        color_array.SetName("Colors")

        for i in range(mapper.GetInput().GetNumberOfPoints()):
            color = [255, 255, 255]

            for tooth_idx in missing_teeth:
                tooth_location_x = 50
                tooth_location_y = 50

                region_x_min, region_x_max = tooth_location_x - 5, tooth_location_x + 5
                region_y_min, region_y_max = tooth_location_y - 5, tooth_location_y + 5

                point = mapper.GetInput().GetPoint(i)
                if region_x_min <= point[0] <= region_x_max and region_y_min <= point[1] <= region_y_max:
                    color = [255, 255, 0]
                    break

            color_array.InsertNextTuple(color)

        mapper.GetInput().GetPointData().SetScalars(color_array)

        self.vtk_render_window.Render()
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

        missing_teeth = [1, 3]
        for tooth_idx in missing_teeth:
            tooth_location_x = 50
            tooth_location_y = 50
            self.ax.scatter(tooth_location_x, tooth_location_y, s=100, c='red', marker='X', label=f"Missing {tooth_idx}")

        for point in self.marking_points:
            self.ax.plot(point[0], point[1], 'ro')

        for marker_position in self.yellow_markers_3d:
            marker = vtk.vtkSphereSource()
            marker.SetCenter(marker_position[0], marker_position[1], marker_position[2])
            marker.SetRadius(5.0)
            marker_mapper = vtk.vtkPolyDataMapper()
            marker_mapper.SetInputConnection(marker.GetOutputPort())
            marker_actor = vtk.vtkActor()
            marker_actor.SetMapper(marker_mapper)
            marker_actor.GetProperty().SetColor(1.0, 1.0, 0.0)
            self.vtk_renderer.AddActor(marker_actor)

        self.ax.legend()
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
                actor.GetProperty().SetOpacity(0.5)
            else:
                actor.GetProperty().SetOpacity(1.0)

            self.vtk_render_window.Render()

    def toggleMarkingMode(self, state):
        self.marking_mode_enabled = state == Qt.Checked

        if not self.marking_mode_enabled:
            self.yellow_markers_3d = []

        self.displayDicomSlice()

    def updateSlice(self):
        self.current_slice = self.slice_slider.value()
        self.displayDicomSlice()
        self.toggleCutoutMode(self.cutout_checkbox.isChecked())

        if self.marking_mode_enabled:
            self.marking_points = self.getMarkingPointsFromUser()

    def getMarkingPointsFromUser(self):
        self.marking_points = []

        def on_click(event):
            if event.inaxes is not None:
                x, y = event.xdata, event.ydata
                self.marking_points.append((x, y))
                self.displayDicomSlice()

        self.canvas.mpl_connect('button_press_event', on_click)
        print("Marking Mode: Click on points to mark. Press 'q' to exit marking mode.")
        plt.show(block=True)
        self.canvas.mpl_disconnect('button_press_event')

        return self.marking_points

    def addYellowMarker3D(self, x, y, z):
        self.yellow_markers_3d.append({'x': x, 'y': y, 'z': z, 'slice': self.current_slice})
        self.displayDicomSlice()

    def segmentTeeth(self):
        segmentation_model = self.initializeSegmentationModel()

        for i, filename in enumerate(self.dicom_files):
            dicom_path = os.path.join(self.directory_path, filename)
            ds = pydicom.read_file(dicom_path)
            pixel_array = ds.pixel_array

            resized_image = cv2.resize(pixel_array, (256, 256))
            normalized_image = resized_image / 255.0
            input_image = np.expand_dims(normalized_image, axis=0)

            segmentation_result = segmentation_model.predict(input_image)
            binary_mask = (segmentation_result > 0.5).astype(np.uint8)

            plt.figure()
            plt.imshow(binary_mask[0, :, :], cmap='gray')
            plt.title(f'Segmentation Result - Slice {i + 1}')
            plt.show()

    def detectDefectiveTeeth(self, pixel_array, slice_index):
        binary_mask = self.segmentTeethForSlice(pixel_array)
        labeled_regions = measure.label(binary_mask)

        for region in measure.regionprops(labeled_regions):
            if region.area > 100:
                centroid = region.centroid
                self.addYellowMarker3D(centroid[1], centroid[0], slice_index)

    def segmentTeethForSlice(self, pixel_array):
        binary_mask = np.zeros_like(pixel_array, dtype=np.uint8)
        binary_mask[pixel_array > 1000] = 1
        return binary_mask


if __name__ == '__main__':
    app = QApplication([])
    dicom_renderer = DicomRenderer()
    dicom_renderer.show()
    app.exec_()
