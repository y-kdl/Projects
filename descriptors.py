from BiT import bio_taxo
from mahotas.features import haralick
from skimage.feature import graycomatrix, graycoprops

import numpy as np

def bitdesc(data):
    all_statistics = bio_taxo(data)
    return all_statistics

def haralick_fct(data):
    all_statistics = haralick(data)
    return all_statistics.flatten()

def glcm(data):
    glcm = graycomatrix(data, [2], [0], 256, symmetric=True, normed=True)
    diss = graycoprops(glcm, 'dissimilarity')[0,0]
    cont = graycoprops(glcm, 'contrast')[0,0]
    corr = graycoprops(glcm, 'correlation')[0,0]
    ener = graycoprops(glcm, 'energy')[0,0]
    homo = graycoprops(glcm, 'homogeneity')[0,0]    
    all_statistics = [diss, cont, corr, ener, homo]
    return all_statistics

def haralick_bit(data):
    bit_features = bitdesc(data)
    haralick_features = haralick_fct(data)
    combined_features = np.concatenate((bit_features, haralick_features), axis=None)
    return combined_features

