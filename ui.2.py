import os
import pydicom
import vtk
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QFileDialog
from PyQt5.QtCore import Qt

class DicomRenderer(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.choose_directory_button = QPushButton('Choose DICOM Directory', self)
        self.choose_directory_button.clicked.connect(self.chooseDirectory)
        self.choose_directory_button.setGeometry(10, 10, 200, 30)

        self.setGeometry(300, 300, 400, 200)
        self.setWindowTitle('DICOM Renderer')

        # VTK rendering components
        self.renderer = vtk.vtkRenderer()
        self.render_window = vtk.vtkRenderWindow()
        self.render_window.SetWindowName("Dental 3D Rendering")
        self.render_window.SetSize(800, 800)
        self.render_window.AddRenderer(self.renderer)

        self.render_window_interactor = vtk.vtkRenderWindowInteractor()
        self.render_window_interactor.SetRenderWindow(self.render_window)

    def loadDicomAndRender(self, directory_path):
        dicom_files = [f for f in os.listdir(directory_path) if f.endswith(".dcm")]
        dicom_files.sort(key=lambda x: int(x.split('Slice')[1].split('.dcm')[0]))

        first_dcm_file = pydicom.read_file(os.path.join(directory_path, dicom_files[0]))
        rows, cols = first_dcm_file.Rows, first_dcm_file.Columns

        volume_data = np.zeros((len(dicom_files), rows, cols), dtype=np.uint16)

        for i, filename in enumerate(dicom_files):
            file_path = os.path.join(directory_path, filename)
            dcm_file = pydicom.read_file(file_path)
            volume_data[i, :, :] = dcm_file.pixel_array

        vtk_volume = vtk.vtkImageData()
        vtk_volume.SetDimensions(cols, rows, len(dicom_files))
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

        self.renderer.AddActor(actor)
        self.renderer.ResetCamera()

        self.render_window.Render()
        self.render_window_interactor.Start()

    def chooseDirectory(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ShowDirsOnly | QFileDialog.DontUseNativeDialog
        directory_path = QFileDialog.getExistingDirectory(self, 'Select DICOM Directory', options=options)

        if directory_path:
            self.loadDicomAndRender(directory_path)

if __name__ == '__main__':
    app = QApplication([])
    dicom_renderer = DicomRenderer()
    dicom_renderer.show()
    app.exec_()
