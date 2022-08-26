import os
from glob import glob

import bios
import numpy as np
import pandas as pd
import radiomics
import SimpleITK as sitk
import tifffile as tfile
from PIL import Image
from radiomics import featureextractor
from tqdm import tqdm


def prepare_paths(data_dir: str):
    '''
    Return paths for the sample dataset.
    '''
    real_img_paths = glob(os.path.join(data_dir, "*_real_voxel.tif"))
    stems = [
        real_img_path[: -len("_real_voxel.tif")] for real_img_path in real_img_paths
    ]
    fake_img_paths = [stem + "_fake_voxel.tif" for stem in stems]
    mask_paths = [stem + "_mask.tif" for stem in stems]
    return real_img_paths, fake_img_paths, mask_paths


def prepare_data_itk(image_paths: list, mask_paths: list):
    '''
    Return lists of images and corresponding masks in SimpleITK format.
    '''
    images = [
        sitk.ReadImage(image_path, imageIO="TIFFImageIO") for image_path in image_paths
    ]
    masks = [
        sitk.ReadImage(mask_path, imageIO="TIFFImageIO") for mask_path in mask_paths
    ]
    return images, masks


def prepare_data(image_paths: list, mask_paths:list):
    '''
    Return lists of images and corresponding masks in numpy format.
    '''
    images = [np.array(tfile.imread(image_path)) for image_path in image_paths]
    masks = [np.array(tfile.imread(mask_path)).astype(int) for mask_path in mask_paths]
    return images, masks


def numpy_to_itk(images):
    '''
    Convert a list of images (or masks) from numpy to SimpleITK format.
    '''
    return [sitk.GetImageFromArray(np.squeeze(img)) for img in images]


def load_features(table_path) -> pd.DataFrame:
    '''
    Load features as pandas table from a tab-separated csv file.
    '''
    return pd.read_csv(table_path, sep="\t")


def get_params_from_yaml(params_path):
    '''
    Read feature extraction parameters from yaml file into python dict.
    '''
    return bios.read(params_path)


# Feature extraction returns OrderedDict which is then converted to pd.DataFrame.
# TODO: Check if numpy dims of the image have to be reversed as in documentation.
# TODO: Check if features need to be returned in individual dataframes
def extract_features(
        params: dict, images: list, masks: list, file_names: list, save_to=None
):
    '''
    Use image segmentations to extract features of the nuclei.
        
        Parameters:
            params (dict): Feature extraction parameters from the parameter file 
            (https://pyradiomics.readthedocs.io/en/latest/customization.html#parameter-file)
           
            images (list[ndarray]): a list of 3D images in numpy ndarray format
           
            masks (list[ndarray]): a list of corresponding 3D masks in numpy ndarray format.

            file_names (list[str]): a list of image names associated with their features.

            save_to (str): a path of a directory where the results are saved.

        Returns:
            df (DataFrame): a table of features with their corresponding images.
    '''
    metrics = []
    labels = []
    extractor = featureextractor.RadiomicsFeatureExtractor()
    extractor._applyParams(paramsDict=params)

    '''Return features in a single dataframe'''
    for i, mask in enumerate(masks):
        print("Image {}/{}".format(i + 1, len(masks)))
        mask_values = np.unique(mask) # Get all nuclei label values 
        with tqdm(total=len(mask_values) - 1, colour="green") as pbar:
            for nucleus_label in mask_values:
                if nucleus_label == 0:
                    continue
                output = extractor.execute(images[i], mask, label=int(nucleus_label))
                values = [
                    float(str(output[k]))
                    for k in output
                    if not k.startswith("diagnostics")
                ]
                if "cols" not in globals():
                    cols = [k for k in output if not k.startswith("diagnostics")]
                values.append(int(nucleus_label))
                values.append(file_names[i])
                metrics.append(values)
                pbar.update(1)
    df = pd.DataFrame(metrics, columns=cols + ["label", "filename"])

    if save_to is not None and merge_features:
        df.to_csv(save_to, sep="\t", encoding="utf-8")
    return df

