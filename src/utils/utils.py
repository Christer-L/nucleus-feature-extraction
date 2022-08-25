import os
from glob import glob

import numpy as np
import pandas as pd
import radiomics
import SimpleITK as sitk
import tifffile as tfile
from radiomics import featureextractor
from tqdm import tqdm


def prepare_paths(data_dir: str):
    real_img_paths = glob(os.path.join(data_dir, "*_real_voxel.tif"))
    stems = [
        real_img_path[: -len("_real_voxel.tif")] for real_img_path in real_img_paths
    ]
    fake_img_paths = [stem + "_fake_voxel.tif" for stem in stems]
    mask_paths = [stem + "_mask.tif" for stem in stems]
    return real_img_paths, fake_img_paths, mask_paths


def prepare_data(image_paths: list, mask_paths: list):
    images = [
        sitk.ReadImage(image_path, imageIO="TIFFImageIO") for image_path in image_paths
    ]
    masks = [
        sitk.ReadImage(mask_path, imageIO="TIFFImageIO") for mask_path in mask_paths
    ]
    return images, masks


def load_features(table_path) -> pd.DataFrame:
    return pd.read_csv(table_path, sep='\t') 


def extract_features(
    params_path, images: list, masks: list, file_names: list, save_to=None
) -> pd.DataFrame:
    metrics = []
    labels = []
    extractor = featureextractor.RadiomicsFeatureExtractor(params_path)
    for i, mask in enumerate(masks):
        print("Image {}/{}".format(i + 1, len(masks)))
        # Get all nuclei label values
        mask_values = np.unique(mask)
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
    if save_to is not None:
        df.to_csv(save_to, sep="\t", encoding="utf-8")
    return df


def get_image_info(path):
    img = sitk.ReadImage(path, imageIO="TIFFImageIO")
    print("Origin: {}, Spacing: {}".format(img.GetOrigin(), img.GetSpacing()))
