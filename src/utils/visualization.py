from string import ascii_letters

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os


def feature_crosscorrelation(features, out):
    """
    Reference: https://seaborn.pydata.org/examples/many_pairwise_correlations.html
    """
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
    plt.savefig(out)

    return corr


def plot_heatmap(corr, out):
    """
    Reference: https://seaborn.pydata.org/examples/many_pairwise_correlations.html
    """
    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))
    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(36, 32))
    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(
        corr,
        mask=mask,
        cmap=cmap,
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

    for tick_label in ax.axes.get_yticklabels():
        if "shape" in str(tick_label):
            tick_label.set_color("blue")
        if "firstorder" in str(tick_label):
            tick_label.set_color("orange")
        if "glcm" in str(tick_label):
            tick_label.set_color("green")
        if "glszm" in str(tick_label):
            tick_label.set_color("purple")
        if "glrlm" in str(tick_label):
            tick_label.set_color("olive")
        if "ngtdm" in str(tick_label):
            tick_label.set_color("pink")
        if "gldm" in str(tick_label):
            tick_label.set_color("cyan")

    for tick_label in ax.axes.get_xticklabels():
        if "shape" in str(tick_label):
            tick_label.set_color("blue")
        if "firstorder" in str(tick_label):
            tick_label.set_color("orange")
        if "glcm" in str(tick_label):
            tick_label.set_color("green")
        if "glszm" in str(tick_label):
            tick_label.set_color("purple")
        if "glrlm" in str(tick_label):
            tick_label.set_color("olive")
        if "ngtdm" in str(tick_label):
            tick_label.set_color("pink")
        if "gldm" in str(tick_label):
            tick_label.set_color("cyan")
            
    ax.set_xticklabels(ax.get_xmajorticklabels(), fontsize=10)
    ax.set_yticklabels(ax.get_ymajorticklabels(), fontsize=10)
    plt.savefig(os.path.join(out, "difference.png"))

    return corr
