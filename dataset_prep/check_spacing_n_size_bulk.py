import os
import pydicom
import numpy as np
import nibabel as nib
import sys
# import DicomNiftiConversion as dnc
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import SimpleITK as sitk
import dateutil
import pandas as pd
sys.path.append("/home/mij17663/code/")
from utils.image_analysis import dicom_utils as du

def check(root_dir):
    for root, dirs, files in os.walk(root_dir):
        dirs.sort()
        for file in files:
            if file.endswith(".dcm"):
                try:
                    ds = pydicom.dcmread(os.path.join(root, file))
                    print(f"Study {root}")
                    print(f"Input {ds.Modality} pixel spacing: {ds.PixelSpacing}, {ds.SliceThickness}")
                    print(f"Input {ds.Modality} image size: {ds.Rows}, {ds.Columns}, {ds.file_meta.FileMetaInformationGroupLength}")
                    # break after processing the first file in the directory
                    break
                except Exception as e:
                    print(f"Error reading DICOM file {file}: {e}")
                    break
        print("-------------------------------------")            

if __name__ == "__main__":
    root_dir = "/Netzwerk-Speicher/ImgRepos/NAC2AC_export/"
    log_file_path = "/home/mij17663/code/petct-img-translation/dataset_prep/check_spacing_n_size_bulk.log"
    # Redirect stdout to a file
    original_stdout = sys.stdout  # Save a reference to the original standard output
    with open(log_file_path, "w") as log_file:
        sys.stdout = log_file  # Change the standard output to the file we created.
        check(root_dir)
        # Reset the standard output to its original value
        sys.stdout = original_stdout

    print(f"Logging complete. Output saved to {log_file_path}")