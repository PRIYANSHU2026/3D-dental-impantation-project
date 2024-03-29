import os
import pydicom
import vtk
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QFileDialog, QVBoxLayout, QHBoxLayout, QSlider
from PyQt5.QtCore import Qt
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

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

        layout = QHBoxLayout(self)

        # Left side for 3D rendering
        vtk_container = QWidget(self)
        vtk_layout = QVBoxLayout(vtk_container)
        vtk_layout.addWidget(self.choose_directory_button)
        vtk_layout.addWidget(self.vtk_render_window_interactor)
        vtk_layout.addWidget(self.slice_slider)
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

        self.canvas.draw()

    def chooseDirectory(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ShowDirsOnly | QFileDialog.DontUseNativeDialog
        self.directory_path = QFileDialog.getExistingDirectory(self, 'Select DICOM Directory', options=options)

        if self.directory_path:
            self.loadDicomAndRender(self.directory_path)
            self.slice_slider.setRange(0, len(self.dicom_files) - 1)
            self.slice_slider.setValue(0)

    def updateSlice(self):
        self.current_slice = self.slice_slider.value()
        self.displayDicomSlice()

if __name__ == '__main__':
    app = QApplication([])
    dicom_renderer = DicomRenderer()
    dicom_renderer.show()
    app.exec_()
