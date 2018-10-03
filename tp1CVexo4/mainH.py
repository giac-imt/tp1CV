import numpy as np
import cv2
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict
