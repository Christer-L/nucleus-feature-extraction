from glob import glob
import os
import tifffile as tfile
import numpy as np
import pandas as pd

import radiomics
from radiomics import featureextractor 

def prepare_paths(data_dir: str):
    real_img_paths = glob(os.path.join(data_dir, "*_real_voxel.tif"))
    stems = [real_img_path[:-len("_real_voxel.tif")] 
            for real_img_path 
            in real_img_paths]
    fake_img_paths = [stem + "_fake_voxel.tif" for stem in stems]
    mask_paths = [stem + "_mask.tif" for stem in stems]

    return real_img_paths, fake_img_paths, mask_paths


def extract_features(
        params_path: str, 
        img_paths: list,
        mask_paths: list,
        file_names: list) -> pd.DataFrame:
    metrics = []
    labels = []
    extractor = featureextractor.RadiomicsFeatureExtractor(params_path)
    for i, mask_path in enumerate(mask_paths):
        mask = tfile.imread(mask_path).astype(int)
        # Get all nuclei label values
        for nucleus_label in np.unique(mask):
            # Ignore the background
            if nucleus_label == 0:
                continue
            output = extractor.execute(
                    img_paths[i], 
                    mask_path, 
                    label=int(nucleus_label))
            values = [float(str(output[k])) 
                    for k in output 
                    if not k.startswith("diagnostics")]
            if 'cols' not in globals():
                cols = [k for k in output if not k.startswith("diagnostics")]
            values.append(int(nucleus_label))
            values.append(file_names[i])
            metrics.append(values)
            print(len(metrics))
        break

    return pd.DataFrame(metrics, 
            columns = cols + ["label", "filename"])
