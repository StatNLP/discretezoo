import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from matplotlib.ticker import PercentFormatter
import seaborn
import math 

from textattack.shared import WordEmbedding

counterfitted_embeddings = WordEmbedding.counterfitted_GLOVE_embedding()
embedding_matrix = counterfitted_embeddings.embedding_matrix

similarity_matrix = np.matmul(embedding_matrix, embedding_matrix.T)
strict_neighbors = similarity_matrix >= 0.9
int_strict_neighbors = np.int32(strict_neighbors)
strict_neighbor_count = np.sum(int_strict_neighbors, axis=-1) - 1 #correct count because everything is self-similar

lax_neighbors = similarity_matrix >= 0.7
int_lax_neighbors = np.int32(lax_neighbors)
lax_neighbor_count = np.sum(int_lax_neighbors, axis=-1) - 1

fig, axs = plt.subplots(2, 2, figsize=[8,8], dpi=300)

# N is the count in each bin, bins is the lower-limit of the bin
N, bins, patches = axs[0][1].hist(strict_neighbor_count, bins=np.linspace(-1, 25, 25), log=True)
# We'll color code by height, but you could use any scalar
fracs = np.log(N + 1) / np.log(N.max() + 1)

# we need to normalize the data to 0..1 for the full range of the colormap
norm = colors.Normalize(fracs.min(), fracs.max())

# Now, we'll loop through our objects and set the color of each accordingly
for thisfrac, thispatch in zip(fracs, patches):
    color = plt.cm.viridis(norm(thisfrac))
    thispatch.set_facecolor(color)

N, bins, patches = axs[0][0].hist(strict_neighbor_count, bins=np.linspace(-1, 25, 25))
fracs = N / N.max()

# we need to normalize the data to 0..1 for the full range of the colormap
norm = colors.Normalize(fracs.min(), fracs.max())

# Now, we'll loop through our objects and set the color of each accordingly
for thisfrac, thispatch in zip(fracs, patches):
    color = plt.cm.viridis(norm(thisfrac))
    thispatch.set_facecolor(color)

N, bins, patches = axs[1][1].hist(lax_neighbor_count, bins=np.linspace(-1, 25, 25), log=True)

fracs = np.log(N + 1) / np.log(N.max() + 1)

# we need to normalize the data to 0..1 for the full range of the colormap
norm = colors.Normalize(fracs.min(), fracs.max())

# Now, we'll loop through our objects and set the color of each accordingly
for thisfrac, thispatch in zip(fracs, patches):
    color = plt.cm.viridis(norm(thisfrac))
    thispatch.set_facecolor(color)

N, bins, patches = axs[1][0].hist(lax_neighbor_count, bins=np.linspace(-1, 25, 25))

fracs = N / N.max()

# we need to normalize the data to 0..1 for the full range of the colormap
norm = colors.Normalize(fracs.min(), fracs.max())

# Now, we'll loop through our objects and set the color of each accordingly
for thisfrac, thispatch in zip(fracs, patches):
    color = plt.cm.viridis(norm(thisfrac))
    thispatch.set_facecolor(color)

axs[0][0].set_ylabel('Cos Sim >= 0.9')
axs[1][0].set_ylabel('Cos Sim >= 0.7')
axs[1][0].set_xlabel('Linear Scale')
axs[1][1].set_xlabel('Log Scale')

fig.supxlabel("Neighborhood Size")
fig.supylabel("Number of Tokens")

plt.savefig('neighborhood_counts_histplot.pdf')