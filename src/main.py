import os

from utils import extract_features, prepare_paths

if __name__ == '__main__':
    params_path = "/home/clohk/projects/nucleus_feature_extraction/src/params.yaml"
    DATA_DIR = "/home/clohk/data/real_and_synthetic_nuclei" 
    img_paths, _, mask_paths = prepare_paths(DATA_DIR) 
    filenames = [os.path.basename(path[0:-14]) for path in img_paths]

    print(extract_features(params_path, img_paths, mask_paths, filenames))
