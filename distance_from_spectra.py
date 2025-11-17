## this script obtains embeddings from spectra and compares the embeddings per file using the Wasserstein
## distance
import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.cluster import MiniBatchKMeans
from scipy.spatial.distance import cdist
import ot ## POT: pip install pot (Python Optimal Transport)
from matchms.importing import load_from_mzml
from lxml import etree

## ms2deepscore contains embedding and model handling functionality:
from ms2deepscore.models import load_model
from ms2deepscore import MS2DeepScore

###############################
## Parameters:
N_CLUSTERS = 100 ## signatures per file (use e.g., 50-200)
EMBEDDING_MODEL_PATH = "data/ms2deepscore_model.pt"
MZML_DIR = "data/"
MAX_SPECTRA_PER_FILE = None  ## sample limit to speed up (None = use all)
###############################

## 1) Load embedding model (pretrained model downloaded via https://zenodo.org/records/13897744)
embedding_model = load_model(EMBEDDING_MODEL_PATH)
m2ds_embeddings = MS2DeepScore(embedding_model)
#w2v = Word2Vec.load(EMBEDDING_MODEL_PATH)
#spec2vec = Spec2Vec(model=w2v, intensity_weighting_power=0.5, allowed_missing_percentage=5.0)

## 2) load mzML files
def extract_polarity_from_mzml(mzml_path):
    """
    Returns a list of 'positive'/'negative'/None per spectrum (MS level 2 only).
    """
    polarity_list = []
    file_mzML = etree.parse(mzml_path)

    for spectrum in file_mzML.findall(".//{*}spectrum"):
        ## obtain MS level
        mslevel = spectrum.find(".//{*}cvParam[@accession='MS:1000511']")
        if mslevel is not None and mslevel.get("value") == "2":
            ## find positive ionization entries
            pos = spectrum.find(".//{*}cvParam[@accession='MS:1000130']")
            ## find negative ionization entries
            neg = spectrum.find(".//{*}cvParam[@accession='MS:1000129']")

            if pos is not None:
                polarity_list.append("positive")
            elif neg is not None:
                polarity_list.append("negative")
            else:
                polarity_list.append(None)
    return polarity_list

def load_spectra_with_polarity(mzml_path):
    spectra = list(load_from_mzml(mzml_path))
    polarities = extract_polarity_from_mzml(mzml_path)

    ## matchms loads only MS2 spectra, lengths should align
    assert len(spectra) == len(polarities)

    for spectrum, polarity in zip(spectra, polarities):
        #if polarity is not None:
        spectrum.set("ionmode", polarity)
        #else:
        #    spectrum.set("ionmode", None)  ## or default
    return spectra

## 3) Convert a matchms Spectrum into an embedding
def get_embedding(spectrum, embedding_model):
    """
    Ms2DeepScore calls embedding_model.get_embedding_array()
    which returns a 1D numpy embedding vector.
    """
    ## obtain the embedding vector
    embedding = embedding_model.get_embedding_array(spectrum)

    ## return as numpy array
    return np.array(embedding)

## 3) For each mzML file: get all embeddings
def get_file_embeddings(mzml_path, embedding_model):
    """
    :param mzml_path: string specifying the path to a mzML file
    :return:
    For each spectra in the file specified by mzml_path, calculate the embedding.
    Return the embedding vectors as a list.
    """
    ## returns list of matchms.Spectrum objects
    spectra = load_spectra_with_polarity(mzml_path)

    ## remove empty spectra
    spectra = [s for s in spectra if s is not None and len(s.peaks.mz) > 0]

    ## obtain embeddings per each spectrum s in spectra
    embeddings = get_embedding(spectra, embedding_model=embedding_model)

    ## return numpy array
    return embeddings

# 4) Compress to K signatures per file
def get_signatures(embeddings, N_CLUSTERS=N_CLUSTERS):
    """
    :param embeddings: numpy array containing the embedding
    :param N_CLUSTERS: integer specifying the number of clusters
    :return:
    For each file, compute k - means signatures(centroids + weights).
    Raw spectra per file can be thousands of points. Comparing every spectrum in file_i
    to every spectrum in file_j would be too expensive (n_i * n_j comparisons).
    to make this tractable, we compress each file’s embedding cloud into K
    representative points (e.g. K = 100). Return the centers and weights of the K representative points.
    """
    ## if fewer spectra than clusters, use spectra themselves as signatures
    if embeddings.shape[0] == 0:
        return np.empty((0, embeddings.shape[1])), np.array([])
    k = min(N_CLUSTERS, embeddings.shape[0])

    ## run MiniBatch k-means on E_i
    kmeans = MiniBatchKMeans(n_clusters=k, random_state=0).fit(embeddings)
    labels = kmeans.fit_predict(embeddings)
    centers = kmeans.cluster_centers_
    counts = np.bincount(labels, minlength=k).astype(float)
    weights = counts / counts.sum()
    ## we now have:
    ## C_i = {c_i1, ci2, ... ciK} cluster centroids in embedding space (shape K * m)
    ## w_i = (w_i1, ..., wiK) relative weights (fraction of spectra per cluster), which sum to 1
    ## So each file is summarized as a discrete probability distribution over K points in embedding space.
    return centers, weights


# 5) Compute pairwise Wasserstein distance using POT (OT with squared Euclidean cost)
# ## each spectrum is a “point” in a high-dimensional “spectral meaning” space,
# ## each file is a “cloud” (distribution) of these points,
# ## the Wasserstein distance between two files measures how much “work” you would need to transform
# ## one point cloud into the other,
# ## so if file_i and file_j have very similar kinds of spectra (many overlapping or nearby embeddings),
# ## their distributions overlap strongly --> small Wasserstein distance.
# ## if file_i and file_j have different sets of compounds (spectra far apart in embedding space) --> large distance
# ## given two files i and j, each has:
# ## K1 cluster centers C_i (centers_i) and weights w_i
# ## K2 cluster centers C_j (centers_j) and weights w_j
# ## we can think of these as two weighted point clouds in R^m:
# ## P_i = {(c_ik, w_ik)}^K_i _k=1
# ## P_j = {(c_jk, w_jk)}^K_j _l=1
def wasserstein_distance(centers_i, weights_i, centers_j, weights_j):
    """
    Compute the 2-Wasserstein distance (with squared Euclidean ground cost) between two
    weighted point clouds using POT (Python Optimal Transport).

    This function treats each file's clustered spectral embeddings as a discrete probability
    distribution in a high-dimensional space. Each cluster centroid acts as a support point,
    and its associated weight represents the probability mass at that point. The Wasserstein
    distance then measures how much "effort" is required to move the probability mass of
    one distribution so that it matches the other.

    Idea begind:
    - Let file i be represented by:
        - Centers: ``centers_i`` of shape (K1, m)
        - Weights: ``weights_i`` of shape (K1,)
      together forming a discrete distribution
      :math:`P_i = {(c_{i,k}, w_{i,k})}_{k=1}^{K_1}`.
    - Let file j be represented analogously by
      :math:`P_j = {(c_{j,l}, w_{j,l})}_{l=1}^{K_2}`.

    The function builds a cost matrix M where:
        M[k, l] = ||c_{i,k} - c_{j,l}||²

    Then it solves the optimal transport (OT) problem using POT’s ``emd2`` function, which
    returns the minimal total transport cost (i.e., the squared 2-Wasserstein distance).
    The square root of this value is returned as the final Wasserstein distance.


    - A small distance indicates that the two files have similar spectral-embedding
      distributions (i.e., many nearby or overlapping cluster centroids).
    - A large distance indicates dissimilar spectra (i.e., clusters far apart in the
      embedding space).
    - ``emd2`` solves the exact OT problem (not entropic regularization).

    :param centers_i : ndarray of shape (K1, m)
        Cluster centroids for distribution i (each row is one centroid in an m-dimensional space).

    :param weights_i : ndarray of shape (K1,)
        Non-negative weights associated with ``centers_i``. Must sum to 1 (POT requirement).

    :param centers_j : ndarray of shape (K2, m)
        Cluster centroids for distribution j.

    :param weights_j : ndarray of shape (K2,)
        Non-negative weights associated with ``centers_j``. Must sum to 1.

    :return: float
        The 2-Wasserstein distance between the two distributions, computed as the
        square root of the optimal transport cost returned by POT’s ``emd2`` solver.

    """
    # build the cost matrix (squared Euclidean):
    ## this gives the pairwise squared Euclidean distances between every cluster centroid in file i
    ## and every cluster centroid in file j,
    ## M is a K_i * K_j matrix with each entry M_kl the "cost" to move a unit of weight from centroid k
    ## in file_i to centroid l in file_j
    M = cdist(centers_i, centers_j, metric="euclidean")**2
    ## solve optimal transport:
    ## this finds the minimal total cost of "moving" the probability mass from P_i to P_j:
    ## w_i, w_j are the weights (probability masses)
    ## M is the cost matrix (distances between support points)
    ## the OT solver finds a transport plan T_kl that minimizes total cost SUM(T_kl M_kl)
    ## POT expects distributions to sum to 1 (they already do)
    ## emd2 returns the transport cost, emd2 is the Waserstein distance squared
    emd2 = ot.emd2(weights_i, weights_j, M)
    return np.sqrt(emd2) ## sqrt of squared cost to get Wasserstein


# 6) Compute distance matrix for all mzML files
# ------------------------------------------------
relative_path = Path(MZML_DIR)
files = list(relative_path.rglob("*.mzML"))
files = sorted([str(f) for f in files])
file_signatures = {}

## precompute embeddings and signatures
for files_i in files:
    print("Processing:", files_i)

    ## obtain new location to store embeddings
    files_i_embedding = ("data/embeddings_MS2/" +
                         files_i.replace("data/", "").replace(".mzML", "") +
                         ".npy")
    ## obtain embeddings and save to the new location, create directory if the path does not exist yet
    embedding_i = get_file_embeddings(files_i, embedding_model=m2ds_embeddings)
    print("Saving embedding to file:", files_i_embedding)
    os.makedirs(os.path.dirname(files_i_embedding), exist_ok=True)
    np.save(files_i_embedding, embedding_i)

    ## obtain signatures and write to file_signatures
    centers, weights = get_signatures(embedding_i, N_CLUSTERS=N_CLUSTERS)
    file_signatures[files_i] = (centers, weights)

## compute full matrix
N = len(files)
D = np.zeros((N, N))

for i in range(N):
    for j in range(i+1, N):
        c_i, w_i = file_signatures[files[i]]
        c_j, w_j = file_signatures[files[j]]
        d = wasserstein_distance(centers_i=c_i, weights_i=w_i, centers_j=c_j, weights_j=w_j)
        D[i, j] = D[j, i] = d

## result: D is file × file distance matrix
df = pd.DataFrame(D, index=file_names, columns=file_names)
df.to_csv("data/ms2deepscore_wasserstein_distance_matrix.csv")
np.save("data/ms2deepscore_signatures_centers_weights.npy", file_signatures)
np.save("data/ms2deepscore_wasserstein_distance_matrix.npy", D)

print("Script completed. Saved Waserstein distances to file data/ms2deepscore_wasserstein_distance_matrix.csv")

