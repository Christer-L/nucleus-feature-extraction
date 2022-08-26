import os
from datetime import datetime

from utils.utils import extract_features, prepare_data, prepare_paths, numpy_to_itk, get_params_from_yaml

PARAMS_PATH = "/home/clohk/projects/nucleus_feature_extraction/src/params.yaml"
DATA_DIR = "/home/clohk/data/real_and_synthetic_nuclei"
TABLE_OUTPUT_DIR = "/home/clohk/projects/nucleus_feature_extraction/results/data/"


if __name__ == "__main__":
    img_paths, synthetic_img_paths, mask_paths = prepare_paths(DATA_DIR)
    filenames = [os.path.basename(path[0:-14]) for path in img_paths]
    real_images, masks = prepare_data(img_paths, mask_paths)

    synthetic_images, masks = prepare_data(synthetic_img_paths, mask_paths)
    synthetic_images = numpy_to_itk(synthetic_images)
    masks = numpy_to_itk(masks)
    real_images, _ = prepare_data(img_paths, mask_paths)
    real_images = numpy_to_itk(real_images)


    now = datetime.now()
    dt_string = now.strftime("%d%m%Y_%H_%M_%S")

    table_file_name_real = os.path.join(
        TABLE_OUTPUT_DIR, "{}-{}.csv".format(dt_string, "real")
    )
    table_file_name_synthetic = os.path.join(
        TABLE_OUTPUT_DIR, "{}-{}.csv".format(dt_string, "synthetic")
    )

    extract_features(get_params_from_yaml(PARAMS_PATH), real_images, masks, filenames, table_file_name_real)
    extract_features(
        get_params_from_yaml(PARAMS_PATH), synthetic_images, masks, filenames, table_file_name_synthetic
    )
