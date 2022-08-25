import os

from utils.utils import load_features
from utils.visualization import get_crosscorrelation

TABLE_INPUT_PATH = "/home/clohk/projects/nucleus_feature_extraction/results/data/"
OUT_PATH = "/home/clohk/projects/nucleus_feature_extraction/results/plots/"

TABLE = "25082022_16_10_48-real"

if __name__ == "__main__":
    # Remove file col from features table
    out_dir = os.path.join(OUT_PATH, TABLE)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    features_path = os.path.join(TABLE_INPUT_PATH, "{}.csv".format(TABLE))
    df_features = load_features(features_path)
    df_features = df_features.loc[:, df_features.columns != 'filename'] 
    df_features = df_features.loc[:, df_features.columns != 'Unnamed: 0'] 
    df_features = df_features.loc[:, df_features.columns != 'label'] 
    get_crosscorrelation(df_features, os.path.join(out_dir, "crosscorr.png"))
    c = df_features.corr()
    s = c.unstack()
    so = s.sort_values(ascending=False)
    df = so.to_frame().reset_index()
    so = df[df['level_0'] != df['level_1']]
    so = df[df['level_0'].str.contains("shape")]
    print(so[:50])





