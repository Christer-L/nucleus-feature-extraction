from string import ascii_letters

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def get_crosscorrelation(features, out, n_top_features=8):
    # Compute the correlation matrix
    corr = features.corr()
    
    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))
    
    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    
    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})

#    plt.matshow(features.corr())
    plt.savefig(out)
#    print("--- Cross-correlation of nucleus features ---")


