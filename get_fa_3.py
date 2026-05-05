from glob import glob
from turtle import position, right
from PIL import Image
import csv
import cv2
import nibabel as nib
import numpy as np
import os
import pandas as pd
import sys
from skimage import measure

def create_directory(directory_path):
    # ディレクトリが存在しない場合に作成
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"ディレクトリ {directory_path} を作成しました。")


def get_nii_fdata(nii_path):
    nii = nib.load(nii_path)
    return nii.get_fdata()

# Function to process each slice
def process_slice(slice_data, slice_index, fa_value_3d, output_path):
    # Threshold the slice to get binary image
    binary_slice = slice_data > 0  # Assuming the bright spots are already binary (1s)

    # Label the regions in the binary image
    labeled_slice = measure.label(binary_slice, connectivity=2)

    # Measure region properties
    regions = measure.regionprops(labeled_slice)

    # Sort regions based on size
    sorted_regions = sorted(regions, key=lambda x: x.area, reverse=True)

    avg_fa_left=0
    avg_fa_right=0
    # Extract the centroids of the two largest regions to determine left and right
    if len(sorted_regions) >= 2:
        region_left = sorted_regions[0] if sorted_regions[0].centroid[1] < sorted_regions[1].centroid[1] else sorted_regions[1]
        region_right = sorted_regions[1] if sorted_regions[0].centroid[1] < sorted_regions[1].centroid[1] else sorted_regions[0]
    
        #画像保存(見てる場所確認)
        fa_normalized=(fa_value_3d*150).astype(np.uint8)
        fa_original = fa_normalized.copy()

        for coord in region_left.coords:
            fa_normalized[coord[0], coord[1], slice_index]=255
        for coord in region_right.coords:
            fa_normalized[coord[0], coord[1], slice_index]=255
        img = Image.fromarray(fa_normalized[:,:,slice_index], mode='L')  # 'L'モードはグレースケール
        ori_img = Image.fromarray(fa_original[:,:,slice_index], mode = 'L')

        ImageWithFA_FolderPath = output_path + f'/ImageWithFA'
        OriginalImageFolderPath = output_path + f'/OriginalImage'

        create_directory(ImageWithFA_FolderPath)
        create_directory(OriginalImageFolderPath)


        img.save(ImageWithFA_FolderPath + f'/image_FA_{slice_index}.png')
        ori_img.save(OriginalImageFolderPath + f'/FA_original_{slice_index}.png')

        # Calculate the average FA values for the left and right regions
        fa_values_left = [fa_value_3d[coord[0], coord[1], slice_index] for coord in region_left.coords]
        fa_values_right = [fa_value_3d[coord[0], coord[1], slice_index] for coord in region_right.coords]
        
        avg_fa_left = np.mean(fa_values_left)
        avg_fa_right = np.mean(fa_values_right)
        
    else:
        avg_fa_left = 0
        avg_fa_right = 0

    threshold = 0.1
    l_flag = 0
    r_flag = 0

    # left_aveがright_aveよりもthresholdの値以上小さい場合
    if avg_fa_left < avg_fa_right - threshold:
        l_flag = 1
    # right_aveがleft_aveよりもthresholdの値以上小さい場合
    elif avg_fa_right < avg_fa_left - threshold:
        r_flag = 1

    # Extract the two largest regions and their coordinates
    slice_data = []
    avg_fa=0
    is_left = True
    for region in sorted_regions[:2]:  # Get only the two largest regions
        flag=0
        for coord in region.coords:  # Iterate over all coordinates in the region
            # z, y, x = slice_index, coord[0], coord[1]
            x, y, z = coord[1], coord[0], slice_index
            fa_value = fa_value_3d[coord[0], coord[1], z]  # Get the corresponding FA value

            difference=abs(avg_fa_left - avg_fa_right)

            #if left
            if x < 128:
                avg_fa=avg_fa_left
                if l_flag == 1:
                    flag=1
                else:
                    flag=0
            #if right
            else:
                avg_fa=avg_fa_right
                if r_flag==1:
                    flag=1
                else:
                    flag=0
            slice_data.append((x, y, z, fa_value, avg_fa, difference, flag))

    return slice_data


# コマンドライン引数からフォルダのパスを取得
if len(sys.argv) < 4:
    sys.exit(1)

folder_path = sys.argv[1]
output_path = folder_path + '/output1013'

nifti_path = folder_path + '/' + sys.argv[2]
roi_path = folder_path + '/' + sys.argv[3]

create_directory(output_path)

# boxel_length = 40
# true_length = 255

roi_arr_3d = get_nii_fdata(roi_path)
fa_arr_3d = get_nii_fdata(nifti_path)

# Initialize list to collect data from all slices
all_data = []

# Let's assume bright spots above a certain threshold represent the regions we're interested in
threshold = roi_arr_3d.max() / 2
roi_arr_3d[roi_arr_3d < threshold] = 0  # Zero out values below threshold to simulate background
roi_arr_3d[roi_arr_3d >= threshold] = 255  # Set bright spots to max value to simulate regions of interest

# Process each slice and collect data
for z in range(roi_arr_3d.shape[2]):  # Iterate over each slice
    #90度回転はしなくても実は揃ってた
    slice_results = process_slice(roi_arr_3d[:, :, z], z, fa_arr_3d, output_path)
    all_data.extend(slice_results)

# Convert all data to a pandas DataFrame
df_all_slices = pd.DataFrame(all_data, columns=['X', 'Y', 'Z', 'FA', 'Avg', 'Difference', 'Flag'])

# Save to a CSV file
csv_path_all_slices = output_path + "/fa_test.csv"
df_all_slices.to_csv(csv_path_all_slices, index=False)

# Output the path to the CSV file
csv_path_all_slices


