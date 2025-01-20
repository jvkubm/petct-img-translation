import os
import pydicom
import numpy as np
import tifffile as tif



def convert_dicom_to_tiff(dicom_dir, output_dir):
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get list of all .dcm files in the directory
    dicom_files = [os.path.join(dicom_dir, f) for f in os.listdir(dicom_dir) if f.endswith('.dcm')]
    
    # Sort the DICOM files by InstanceNumber or SliceLocation
    dicom_files.sort(key=lambda x: int(pydicom.dcmread(x).InstanceNumber))

    # Read the first DICOM file to get the dimensions
    ds = pydicom.dcmread(dicom_files[0])
    num_slices = len(dicom_files)
    rows, cols = ds.Rows, ds.Columns

    # Create a 3D numpy array to hold the pixel data
    volume = np.zeros((num_slices, rows, cols), dtype=ds.pixel_array.dtype)

    # Read each DICOM file and store the pixel data in the 3D array
    for i, dicom_file in enumerate(dicom_files):
        ds = pydicom.dcmread(dicom_file)
        volume[i, :, :] = ds.pixel_array


    study_no = dicom_dir.split('/')[-3]
    # Save each slice as a TIFF image
    for i in range(num_slices):
        tif.imwrite(os.path.join(output_dir, f'{study_no}_slice_{i:03d}.tiff'), volume[i, :, :])

    print(f"Conversion complete for {dicom_dir}. {num_slices} slices saved to {output_dir}")

# Example usage
dicom_dir = '/Netzwerk-Speicher/ImgRepos/NAC2AC_export/Nac2Ac0001/Pet_Pet_Fdg_Ld_Wb_Cbm_(Adult) - 1/PET_WB_6'
output_dir = '/mnt/data/mij17663/nac2ac_sep/2d_slices/AC_PET'
# convert_dicom_to_tiff(dicom_dir, output_dir)

def go_through_dirs(root_dir):
    for root, dirs, files in os.walk(root_dir):
        for d in dirs:
            if d.startswith('PET_WB_Uncorrected'):
                dicom_dir = os.path.join(root, d)
                output_dir = '/mnt/data/mij17663/nac2ac_sep/2d_slices/NAC_PET/'
                convert_dicom_to_tiff(dicom_dir, output_dir)
            elif d.startswith('PET_WB'):
                dicom_dir = os.path.join(root, d)
                output_dir = '/mnt/data/mij17663/nac2ac_sep/2d_slices/AC_PET/'
                convert_dicom_to_tiff(dicom_dir, output_dir)

root_dir = '/Netzwerk-Speicher/ImgRepos/NAC2AC_export/'
go_through_dirs(root_dir)