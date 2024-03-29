import os
import pydicom
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QFileDialog, QVBoxLayout, QHBoxLayout, QSlider, QCheckBox
from PyQt5.QtCore import Qt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import cv2

class DicomRenderer(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.choose_directory_button = QPushButton('Choose DICOM Directory', self)
        self.choose_directory_button.clicked.connect(self.chooseDirectory)

        self.figure_axial, self.ax_axial = plt.subplots()
        self.canvas_axial = FigureCanvas(self.figure_axial)
        self.toolbar_axial = NavigationToolbar(self.canvas_axial, self)

        self.figure_coronal, self.ax_coronal = plt.subplots()
        self.canvas_coronal = FigureCanvas(self.figure_coronal)
        self.toolbar_coronal = NavigationToolbar(self.canvas_coronal, self)

        self.slice_slider = QSlider(Qt.Horizontal)
        self.slice_slider.valueChanged.connect(self.updateSlice)

        self.brightness_slider = QSlider(Qt.Horizontal)
        self.brightness_slider.setMinimum(-100)
        self.brightness_slider.setMaximum(100)
        self.brightness_slider.setValue(0)
        self.brightness_slider.valueChanged.connect(self.updateBrightness)

        self.cutout_checkbox = QCheckBox("Mesh Cut-Out Mode")
        self.cutout_checkbox.stateChanged.connect(self.toggleCutoutMode)

        self.marking_checkbox = QCheckBox("Marking Mode")
        self.marking_checkbox.stateChanged.connect(self.toggleMarkingMode)

        layout = QHBoxLayout(self)

        vtk_container = QWidget(self)
        vtk_layout = QVBoxLayout(vtk_container)
        vtk_layout.addWidget(self.choose_directory_button)
        vtk_layout.addWidget(self.slice_slider)
        vtk_layout.addWidget(self.brightness_slider)
        vtk_layout.addWidget(self.cutout_checkbox)
        vtk_layout.addWidget(self.marking_checkbox)
        layout.addWidget(vtk_container)

        matplotlib_container_axial = QWidget(self)
        matplotlib_layout_axial = QVBoxLayout(matplotlib_container_axial)
        matplotlib_layout_axial.addWidget(self.toolbar_axial)
        matplotlib_layout_axial.addWidget(self.canvas_axial)
        layout.addWidget(matplotlib_container_axial)

        matplotlib_container_coronal = QWidget(self)
        matplotlib_layout_coronal = QVBoxLayout(matplotlib_container_coronal)
        matplotlib_layout_coronal.addWidget(self.toolbar_coronal)
        matplotlib_layout_coronal.addWidget(self.canvas_coronal)
        layout.addWidget(matplotlib_container_coronal)

        self.setGeometry(300, 300, 800, 400)
        self.setWindowTitle('DICOM Renderer')

        self.dicom_files = []
        self.current_slice = 0
        self.directory_path = ""
        self.marking_points = []
        self.marking_mode_enabled = False
        self.yellow_markers_3d = []

    def loadDicomAndRender(self, directory_path):
        self.dicom_files = [f for f in os.listdir(directory_path) if f.endswith(".dcm")]
        self.dicom_files.sort(key=lambda x: int(x.split('Slice')[1].split('.dcm')[0]))

        first_dcm_file = pydicom.read_file(os.path.join(directory_path, self.dicom_files[0]))
        self.rows, self.cols = first_dcm_file.Rows, first_dcm_file.Columns

        volume_data = np.zeros((len(self.dicom_files), self.rows, self.cols), dtype=np.uint16)

        for i, filename in enumerate(self.dicom_files):
            file_path = os.path.join(directory_path, filename)
            dcm_file = pydicom.read_file(file_path)
            volume_data[i, :, :] = dcm_file.pixel_array

        self.volume_data = volume_data

        self.axial_image = self.ax_axial.imshow(self.volume_data[self.current_slice], cmap='gray', aspect='auto')
        self.ax_axial.set_title(f'DICOM Axial Slice {self.current_slice + 1}/{len(self.dicom_files)}')
        self.ax_axial.set_axis_off()

        self.coronal_slice = np.transpose(self.volume_data[:, self.current_slice, :], (1, 0))
        self.coronal_image = self.ax_coronal.imshow(self.coronal_slice, cmap='gray', aspect='auto')
        self.ax_coronal.set_title(f'DICOM Coronal Slice {self.current_slice + 1}/{len(self.dicom_files)}')
        self.ax_coronal.set_axis_off()

        self.canvas_axial.draw_idle()
        self.canvas_coronal.draw_idle()

        self.slice_slider.setRange(0, len(self.dicom_files) - 1)
        self.slice_slider.setValue(0)

    def chooseDirectory(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ShowDirsOnly | QFileDialog.DontUseNativeDialog
        self.directory_path = QFileDialog.getExistingDirectory(self, 'Select DICOM Directory', options=options)

        if self.directory_path:
            self.loadDicomAndRender(self.directory_path)

    def toggleCutoutMode(self, state):
        pass

    def toggleMarkingMode(self, state):
        pass

    def updateSlice(self):
        self.current_slice = self.slice_slider.value()
        img = self.volume_data[self.current_slice]
        img = self.clahe_equalization(img)
        img = self.adjustBrightness(img, self.brightness_slider.value())
        self.axial_image.set_array(img)

        coronal_img = self.volume_data[:, self.current_slice, :]
        coronal_img = self.clahe_equalization(coronal_img)
        coronal_img = self.adjustBrightness(coronal_img, self.brightness_slider.value())
        coronal_img = np.transpose(coronal_img, (1, 0))
        self.coronal_image.set_array(coronal_img)

        self.ax_axial.set_title(f'DICOM Axial Slice {self.current_slice + 1}/{len(self.dicom_files)}')
        self.ax_coronal.set_title(f'DICOM Coronal Slice {self.current_slice + 1}/{len(self.dicom_files)}')

        self.canvas_axial.draw_idle()
        self.canvas_coronal.draw_idle()

    def adjustBrightness(self, img, brightness):
        # Clip the brightness value to avoid overflow
        brightness = np.clip(brightness, -100, 100)

        # Add brightness to the image
        img = img + brightness

        # Clip the image pixel values to ensure they remain in the valid range
        img = np.clip(img, 0, 65535)

        return img

    def clahe_equalization(self, img):
        # Convert to 8-bit unsigned integer
        img_uint8 = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        # Apply CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_equalized = clahe.apply(img_uint8)

        return img_equalized

    def updateBrightness(self):
        self.updateSlice()

if __name__ == '__main__':
    app = QApplication([])
    dicom_renderer = DicomRenderer()
    dicom_renderer.show()
    app.exec_()
