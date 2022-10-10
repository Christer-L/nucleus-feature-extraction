from enum import Enum
import math
import os
from glob import glob
from typing import List, Optional

import numpy as np
import pandas as pd
import radiomics
import SimpleITK as sitk
from radiomics import featureextractor
from tqdm import tqdm
from arkitekt.apps.connected import ConnectedApp
from mikro.api.schema import (
    TableFragment,
    RepresentationFragment,
    from_df,
    OmeroFileFragment,
    upload_bioimage,
    create_feature,
    create_label,
    get_representation,
)
import yaml
import seaborn as sns
import matplotlib.pyplot as plt
from rekuest.utils import progress


app = ConnectedApp()
app.fakts.grant.open_browser = False
app.fakts.grant.name = "Radiomics"
app.fakts.grant.discovery.base_url = "http://herre:8000/f/"


def prepare_paths(data_dir: str):
    """
    Return paths for the sample dataset.
    """
    real_img_paths = glob(os.path.join(data_dir, "*_real_voxel.tif"))
    stems = [
        real_img_path[: -len("_real_voxel.tif")] for real_img_path in real_img_paths
    ]
    fake_img_paths = [stem + "_fake_voxel.tif" for stem in stems]
    mask_paths = [stem + "_mask.tif" for stem in stems]
    return real_img_paths, fake_img_paths, mask_paths


def prepare_data_itk(image_paths: list, mask_paths: list):
    """
    Return lists of images and corresponding masks in SimpleITK format.
    """
    images = [
        sitk.ReadImage(image_path, imageIO="TIFFImageIO") for image_path in image_paths
    ]
    masks = [
        sitk.ReadImage(mask_path, imageIO="TIFFImageIO") for mask_path in mask_paths
    ]
    return images, masks


def prepare_data(image_paths: list, mask_paths: list):
    """
    Return lists of images and corresponding masks in numpy format.
    """
    images = [np.array(tfile.imread(image_path)) for image_path in image_paths]
    masks = [np.array(tfile.imread(mask_path)).astype(int) for mask_path in mask_paths]
    return images, masks


def numpy_to_itk(images):
    """
    Convert a list of images (or masks) from numpy to SimpleITK format.
    """
    return [sitk.GetImageFromArray(np.squeeze(img)) for img in images]


def load_features(table_path) -> pd.DataFrame:
    """
    Load features as pandas table from a tab-separated csv file.
    """
    return pd.read_csv(table_path, sep="\t")


def get_params_from_yaml(params_path):
    """
    Read feature extraction parameters from yaml file into python dict.
    """
    return bios.read(params_path)


# Feature extraction returns OrderedDict which is then converted to pd.DataFrame.
# TODO: Check if numpy dims of the image have to be reversed as in documentation.
# TODO: Check if features need to be returned in individual dataframes


class RadiomicsFeatures(str, Enum):
    Autocorrelation = "Autocorrelation"
    JointAverage = "JointAverage"
    ClusterProminence = "ClusterProminence"
    ClusterShade = "ClusterShade"
    ClusterTendency = "ClusterTendency"
    Contrast = "Contrast"
    Correlation = "Correlation"


def extract_features_u(
    params: dict,
    image: RepresentationFragment,
    mask: RepresentationFragment,
    save_to=None,
    voxel_threshold=10,
    merge_features=False,
):
    """
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
    """

    image_data = image.data.sel(c=0, t=0).data.compute()
    mask_data = mask.data.sel(c=0, t=0).data.compute()

    raw_image = sitk.GetImageFromArray(image_data)
    mask_image = sitk.GetImageFromArray(mask_data)

    metrics = []
    extractor = featureextractor.RadiomicsFeatureExtractor()
    extractor._applyParams(paramsDict=params)

    """Return features in a single dataframe"""
    mask_values, mask_counts = np.unique(
        mask_data, return_counts=True
    )  # Get all nuclei label values
    print(mask_values, mask_counts)
    print(image_data, mask_data)
    with tqdm(total=len(mask_values) - 1, colour="green") as pbar:
        for index, nucleus_label in enumerate(mask_values):
            if mask_counts[index] > voxel_threshold:
                if nucleus_label == 0:
                    continue
                output = extractor.execute(
                    raw_image, mask_image, label=int(nucleus_label)
                )
                values = [
                    float(str(output[k]))
                    for k in output
                    if not k.startswith("diagnostics")
                ]
                if "cols" not in globals():
                    cols = [k for k in output if not k.startswith("diagnostics")]
                values.append(int(nucleus_label))
                metrics.append(values)
                progress(math.floor(index / len(mask_values) * 100))
                pbar.update(1)

    df = pd.DataFrame(metrics, columns=cols + ["label"])

    if save_to is not None and merge_features:
        df.to_csv(save_to, sep="\t", encoding="utf-8")
    return df


@app.rekuest.register()
def extract_feature_basic(
    mask: RepresentationFragment,
    image: Optional[RepresentationFragment],
    voxel_threshold: int = 10,
    calc_features: List[RadiomicsFeatures] = [
        RadiomicsFeatures.Autocorrelation,
        RadiomicsFeatures.ClusterProminence,
    ],
) -> TableFragment:
    """Exract features

    Extract features from a given image and mask.

    Args:
        mask (RepresentationFragment): _description_
        image (RepresentationFragment): _description_

    Returns:
        TableFragment: _description_
    """

    with open("params.yaml") as f:
        params = yaml.safe_load(f)

    image = image or get_representation(mask.origins[0])

    raw_image = image
    mask_image = mask

    df = extract_features_u(
        params,
        raw_image,
        mask_image,
        voxel_threshold=voxel_threshold,
    )

    # extract_features
    return from_df(df, name=f"Features of {image.name}")


@app.rekuest.register()
def feature_crosscorrelation(table: TableFragment) -> OmeroFileFragment:
    """Feature cross

    Displays the features of the crosscorrelation

    Args:
        table (TableFragment): _description_

    Returns:
        OmeroFile: _description_
    """

    features = table.data

    # Compute the correlation matrix

    corr = features.corr()
    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))
    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(32, 32))
    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(
        corr,
        mask=mask,
        cmap=cmap,
        vmax=1.0,
        vmin=-1.0,
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.5},
        xticklabels=True,
        yticklabels=True,
    )

    # use matplotlib.colorbar.Colorbar object
    cbar = ax.collections[0].colorbar
    # here set the labelsize by 20
    cbar.ax.tick_params(labelsize=20)

    ax.set_xticklabels(ax.get_xmajorticklabels(), fontsize=10)
    ax.set_yticklabels(ax.get_ymajorticklabels(), fontsize=10)
    plt.savefig("test.png")

    return upload_bioimage(open("test.png", "rb"), name="test.png")


with app:
    app.rekuest.run()
