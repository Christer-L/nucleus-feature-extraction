import os
from datetime import datetime

from utils import extract_features, prepare_data, prepare_paths

if __name__ == '__main__':
    PARAMS_PATH = "/home/clohk/projects/nucleus_feature_extraction/src/params.yaml"
    DATA_DIR = "/home/clohk/data/real_and_synthetic_nuclei" 
    TABLE_OUTPUT_DIR = "/home/clohk/projects/nucleus_feature_extraction/results/data/" 
    img_paths, _, mask_paths = prepare_paths(DATA_DIR) 
    filenames = [os.path.basename(path[0:-14]) for path in img_paths]
    images, masks = prepare_data(img_paths, mask_paths)
    experiment_name = "test1"

    now = datetime.now()
    dt_string = now.strftime("%d%m%Y_%H_%M_%S")
    table_file_name = os.path.join(
            TABLE_OUTPUT_DIR,
            "{}-{}.csv".format(dt_string, experiment_name))

    print(extract_features(PARAMS_PATH, images, masks, filenames, table_file_name))
