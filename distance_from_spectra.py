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
# Choose embedding package:
# Option 1: spec2vec
from gensim.models import Word2Vec
from spec2vec import Spec2Vec
from spec2vec.vector_operations import calc_vector
# Option 2 (alternative): from ms2deepscore.embedding_creator import EmbeddingCreator
from ms2deepscore.models import load_model
# Use whichever you installed

###############################
## Parameters you can tune
N_CLUSTERS = 100 ## signatures per file (use 50-200)
N_JOBS = 8 ## parallel workers
EMBEDDING_MODEL_PATH = "spec2vec_pretrained.model" ## or ms2deepscore model
model_file_name = "data/ms2deepscore_model.pt"
model = load_model(model_file_name)
MZML_DIR = "data/"
MAX_SPECTRA_PER_FILE = None  ## sample limit to speed up (None = use all)
###############################

## 1) Load embedding model (spec2vec example)
w2v = Word2Vec.load(EMBEDDING_MODEL_PATH)
spec2vec = Spec2Vec(model=w2v, intensity_weighting_power=0.5, allowed_missing_percentage=5.0)

def read_spectra(mzml_path):
    ## returns list of matchms.Spectrum objects
    return list(load_from_mzml(mzml_path))

def spectrum_to_embedding(spectrum):
    ## calc_vector returns numpy array
    vec = calc_vector(spectrum, spec2vec)
    return np.array(vec, dtype=float)

def file_embeddings(mzml_path):#, max_spec=None):
    specs = read_spectra(mzml_path)
    ## downsample if max_spec is not None and there are more spectra than max_spec
    if max_spec is not None and len(specs) > max_spec:
        ## simple downsample
        idx = np.linspace(0, len(specs)-1, max_spec).astype(int)
        specs = [specs[i] for i in idx]

    ## obtain the embeddings for each spectrum in specs
    embs = [spectrum_to_embedding(s) for s in specs]
    embs = [e for e in embs if e is not None and np.isfinite(e).all()]
    if len(embs)==0:
        return np.empty((0, w2v.vector_size))
    return np.vstack(embs)

## 2) For each file, compute k-means signatures (centroids + weights)
## Raw spectra per file can be thousands of points. Comparing every spectrum in file_i
## to every spectrum in file_j would be too expensive (n_i * n_j comparisons).
## to make this tractable, we compress each file’s embedding cloud into K
## representative points (e.g. K = 100):
def file_signatures(embeddings, n_clusters=N_CLUSTERS):
    if embeddings.shape[0] == 0:
        return np.empty((0, embeddings.shape[1])), np.array([])
    ## if fewer spectra than clusters, use spectra themselves as signatures
    k = min(n_clusters, embeddings.shape[0])
    ## run MiniBatch k-means on E_i
    kmeans = MiniBatchKMeans(n_clusters=k, batch_size=1000, random_state=0)
    labels = kmeans.fit_predict(embeddings)
    centers = kmeans.cluster_centers_
    counts = np.bincount(labels, minlength=k).astype(float)
    weights = counts / counts.sum()
    ## we now have:
    ## C_i = {c_i1, ci2, ... ciK} cluster centroids in embedding space (shape K * m)
    ## w_i = (w_i1, ..., wiK) relative weights (fraction of spectra per cluster), which sum to 1
    ## So each file is summarized as a discrete probability distribution over K points in embedding space.
    return centers, weights

## 3) Load all files and produce centers and weights from embedings
relative_path = Path(MZML_DIR)
files = list(relative_path.rglob("*.mzML"))
files = sorted([str(f) for f in files])
##files = sorted([os.path.join(MZML_DIR, f) for f in os.listdir(MZML_DIR) if f.endswith(".mzML")])
file_centers = []
file_weights = []
file_names = []

for fpath in files:
    embs = file_embeddings(fpath, max_spec=MAX_SPECTRA_PER_FILE)
    centers, weights = file_signatures(embs, n_clusters=N_CLUSTERS)
    file_centers.append(centers) ## shape (k, d)
    file_weights.append(weights) ## shape (k,)
    file_names.append(os.path.basename(fpath))

## 4) Compute pairwise Wasserstein distance using POT (OT with squared Euclidean cost)
## each spectrum is a “point” in a high-dimensional “spectral meaning” space,
## each file is a “cloud” (distribution) of these points,
## the Wasserstein distance between two files measures how much “work” you would need to transform
## one point cloud into the other,
## so if file_i and file_j have very similar kinds of spectra (many overlapping or nearby embeddings),
## their distributions overlap strongly --> small Wasserstein distance.
## if file_i and file_j have different sets of compounds (spectra far apart in embedding space) --> large distance
## given two files i and j, each has:
## K1 cluster centers C_i (centers_i) and weights w_i
## K2 cluster centers C_j (centers_j) and weights w_j
## we can think of these as two weighted point clouds in R^m:
## P_i = {(c_ik, w_ik)}^K_i _k=1
## P_j = {(c_jk, w_jk)}^K_j _l=1
def wasserstein_between(centers_i, w_i, centers_j, w_j):
    ## if empty cases
    if centers_i.shape[0] == 0 and centers_j.shape[0] == 0:
        return 0.0
    if centers_i.shape[0] == 0 or centers_j.shape[0] == 0:
        return np.inf
    ## build the cost matrix (squared Euclidean):
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
    emd2 = ot.emd2(w_i, w_j, M)
    return np.sqrt(emd2) ## sqrt of squared cost to get Wasserstein

## pairwise
## do the calculation for all file pairs i, j, resulting in a symmetric distance matrix D of shape
## (num_files * num_files)
n = len(files)
D = np.zeros((n, n))
for i in range(n):
    for j in range(i, n):
        d = wasserstein_between(file_centers[i], file_weights[i], file_centers[j], file_weights[j])
        D[i, j] = D[j, i] = d

df = pd.DataFrame(D, index=file_names, columns=file_names)
df.to_csv("wasserstein_distances.csv")
print("Script completed. Saved Waserstein distances to file wasserstein_distances.csv")