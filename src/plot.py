import os

from utils.utils import load_features
from utils.visualization import feature_crosscorrelation, plot_heatmap

TABLE_INPUT_PATH = "/home/clohk/projects/nucleus_feature_extraction/results/data/"
OUT_PATH = "/home/clohk/projects/nucleus_feature_extraction/results/plots/"

TABLE_REAL = "26082022_13_29_38-real"
TABLE_SYNTHETIC = "26082022_13_29_38-synthetic"

if __name__ == "__main__":
    # Remove file col from features table
    out_dir = os.path.join(OUT_PATH, TABLE_REAL)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    features_path = os.path.join(TABLE_INPUT_PATH, "{}.csv".format(TABLE_REAL))
    df_features = load_features(features_path)
    df_features = df_features.loc[:, df_features.columns != "filename"]
    df_features = df_features.loc[:, df_features.columns != "Unnamed: 0"]
    df_features = df_features.loc[:, df_features.columns != "label"]

    features_path = os.path.join(TABLE_INPUT_PATH, "{}.csv".format(TABLE_SYNTHETIC))
    df_features_s = load_features(features_path)
    df_features_s = df_features_s.loc[:, df_features_s.columns != "filename"]
    df_features_s = df_features_s.loc[:, df_features_s.columns != "Unnamed: 0"]
    df_features_s = df_features_s.loc[:, df_features_s.columns != "label"]

    # Fix difference
    plot_heatmap(abs(df_features.corr() - df_features_s.corr()), out_dir)

    c = df_features.corr()
    s = c.unstack()
    so = s.sort_values(ascending=False)
    df = so.to_frame().reset_index()
    so = df[df["level_0"] != df["level_1"]]
    so = df[df["level_0"].str.contains("shape")]

    print(so[:50])
