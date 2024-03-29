import os
import pydicom
import vtk
import numpy as np
from tkinter import Tk, filedialog, Button, Label, Entry

def load_dicom_and_render(directory_path, num_files):
    # Collect all DICOM files in the directory
    dicom_files = [f for f in os.listdir(directory_path) if f.endswith(".dcm")]

    # Sort the files based on their numerical order (assuming filenames contain numbers)
    dicom_files.sort(key=lambda x: int(x.split('Slice')[1].split('.dcm')[0]))

    # Use the specified number of files or all available files if num_files is greater than the total number of files
    dicom_files = dicom_files[:num_files]

    # ... (rest of the code remains the same)

def choose_directory_and_num_files():
    root = Tk()
    root.withdraw()  # Hide the main window

    # Create a window for choosing directory and specifying the number of files
    subwindow = Tk()
    subwindow.title("Choose Directory and Number of Files")

    directory_label = Label(subwindow, text="Select DICOM Directory:")
    directory_label.grid(row=0, column=0, padx=10, pady=10)

    directory_path = filedialog.askdirectory(title="Select DICOM Directory")
    directory_entry = Entry(subwindow, width=40)
    directory_entry.insert(0, directory_path)
    directory_entry.grid(row=0, column=1, padx=10, pady=10)

    num_files_label = Label(subwindow, text="Number of DICOM Files:")
    num_files_label.grid(row=1, column=0, padx=10, pady=10)

    num_files_entry = Entry(subwindow, width=10)
    num_files_entry.insert(0, "10")  # Default value, you can change it as needed
    num_files_entry.grid(row=1, column=1, padx=10, pady=10)

    def load_and_render():
        subwindow.destroy()
        load_dicom_and_render(directory_entry.get(), int(num_files_entry.get()))

    load_button = Button(subwindow, text="Load and Render", command=load_and_render)
    load_button.grid(row=2, columnspan=2, pady=20)

    subwindow.mainloop()

# Create a button to choose the directory and specify the number of files
choose_directory_button = Button(
    text="Choose DICOM Directory and Number of Files",
    command=choose_directory_and_num_files
)
choose_directory_button.pack()

# Main Tkinter loop
Tk().mainloop()
