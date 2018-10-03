import numpy as np
import cv2
from sklearn.neighbors import KNeighborsClassifier
from IPython.display import display


def unpickle_doc(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def unpickle_tp(file):
    # Désérialiser les fichiers image afin de permettre l’accès aux données et aux labels:
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo)
    return dict


def default_label_fn(i, original):
    return original


def show_img(img_arr, label_arr, meta, index, label_fn=default_label_fn):
    # Given a numpy array of image from CIFAR-10 labels this method transform the data so that PIL can read
    # and show the image.Check here how CIFAR encodes the image http://www.cs.toronto.edu/~kriz/cifar.html
    one_img = img_arr[index, :]
    # Assume image size is 32 x 32. First 1024 px is r, next 1024 px is g, last 1024 px is b from the (r,g b)
    # channel
    r = one_img[:1024].reshape(32, 32)
    g = one_img[1024:2048].reshape(32, 32)
    b = one_img[2048:].reshape(32, 32)
    rgb = np.dstack([r, g, b])
    #img = Image.fromarray(np.array(rgb), 'RGB')
    img = cv2.imread('code-route.jpg')
    # display(img)
    print(label_fn(index, meta[label_arr[index][0]].decode('utf-8')))


def pred_label_fn(i, original):
    return original + '::' + meta[YPred[i]].decode('utf-8')


basedir_data = "./data/"
rel_path = basedir_data + "cifar-10-batches-py/"

X = unpickle_doc(rel_path + 'data_batch_1')
img_data = X[b'data']
img_label_orig = img_label = X[b'labels']
img_label = np.array(img_label).reshape(-1, 1)

test_X = unpickle_doc(rel_path + 'test_batch');
test_data = test_X[b'data']
test_label = test_X[b'labels']
test_label = np.array(test_label).reshape(-1, 1)

sample_img_data = img_data[0:10, :]
# print(sample_img_data)
# print('shape', sample_img_data.shape)

batch = unpickle_doc(rel_path + 'batches.meta');
meta = batch[b'label_names']
# print(meta)

data_point_no = 10
sample_test_data = test_data[:data_point_no, :]
nbrs = KNeighborsClassifier(n_neighbors=3, algorithm='brute').fit(img_data, img_label_orig)
YPred = nbrs.predict(sample_test_data)
for i in range(0, len(YPred)):
    show_img(sample_test_data, test_label, meta, i, label_fn=pred_label_fn)
