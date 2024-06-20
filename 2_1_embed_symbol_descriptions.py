"""
This script takes the descriptions of the german traffic signs in the
descriptions.py module and embeds them using SBERT.

These embeddings are saved for later use when learning the symbol domain.
"""

# ------------------------------------------------------------------------------
# imports

import numpy as np
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
from modules.symbol_descriptions import descriptions

# ------------------------------------------------------------------------------

def main():

    SAVE_PATH = 'local/models/symbol_embeddings.npy'
    DESCRIPTIONS = list(descriptions.values())

    model = SentenceTransformer('all-mpnet-base-v2')
    embeddings = model.encode(DESCRIPTIONS)

    np.save(SAVE_PATH, embeddings)
    return embeddings

# ------------------------------------------------------------------------------

if __name__ == "__main__":

    EVALUATE = False    # plots similarity/distance heatmaps if True

    embeddings = main()

    if EVALUATE:
        distances = np.empty((len(embeddings), len(embeddings)))
        for i, e1 in enumerate(embeddings):
            for j, e2 in enumerate(embeddings):
                distances[i, j] = np.linalg.norm(e1 - e2)
        similarities = np.exp(-1.0*distances**2)
        cos_sims = np.dot(embeddings, embeddings.T)

        fig, axs = plt.subplots(1, 3, figsize=(16, 4))
        cax0 = axs[0].imshow(distances, cmap='hot', interpolation='nearest')
        axs[0].set_title('Euclidean distances')
        cax1 = axs[1].imshow(similarities, cmap='hot', interpolation='nearest')
        axs[1].set_title('Similarities')
        cax2 = axs[2].imshow(cos_sims, cmap='hot', interpolation='nearest')
        axs[2].set_title('Cosine similarities')
        fig.colorbar(cax0, ax=axs[0], orientation='vertical')
        fig.colorbar(cax1, ax=axs[1], orientation='vertical')
        fig.colorbar(cax2, ax=axs[2], orientation='vertical')
        plt.show()

# ------------------------------------------------------------------------------