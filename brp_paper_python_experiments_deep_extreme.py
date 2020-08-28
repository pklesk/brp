# title           :brp_python_implementation.py
# description     :Example Python implementation of Boosted Random Planes algorithm (and variants), accompanying research paper: "Can Boosted Randomness Mimic Learning Algorithms of Geometric Nature? Example of a Simple Algorithm that Converges in Probability to Hard-Margin SVM". 
# author          :Marcin Korzeń, Przemysław Klęsk
# date            :20200827
# version         :1.1
# license         :CC-BY 4.0
# acks            :This work was financed by the National Science Centre, Poland. Research project no.: 2016/21/B/ST6/01495.

import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.sparse as sp
import time
import os
import zipfile

from keras.models import Sequential
from keras.layers import Dense, Flatten, Reshape, ActivityRegularization
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D, AveragePooling2D, GlobalMaxPooling2D
from keras import optimizers
from keras.utils import plot_model
from keras import layers
from keras import regularizers

import tensorflow

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import train_test_split
from sklearn.utils import check_random_state
from sklearn import metrics
from sklearn.metrics import confusion_matrix, accuracy_score
    
from sklearn.preprocessing import StandardScaler
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier

from brp_python_implementation import BoostedRandomPlanes, BoostedRandomPlanesFast, BoostedRandomPlanesLogits, SvmCxopt, LinearSklearn

import keras
from scipy.interpolate import dfitpack
import sklearn

def load_mnist(train_samples=1000, test_samples=10000):
    rnd = np.random.RandomState(0)
    # Load data from https://www.openml.org/d/554
    # X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
    # with open('mnist_784.npy', 'wb') as f:
    #     pickle.dump( (X,y), f)
    
    if not os.path.exists("data/mnist_784.npy"):     
        print("UNZIPPING MNIST DATA... [data/mnist_784.zip]")   
        with zipfile.ZipFile("data/mnist_784.zip", "r") as zip_ref:
            zip_ref.extractall("data")
        print("UNZIPPING MNIST DATA DONE.")
    
    with open("data/mnist_784.npy", 'rb') as f:
        X, y = pickle.load(f)
    yy = y.copy()
    yy = yy.astype(int)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, yy, train_size=train_samples, test_size=test_samples, random_state=rnd)
    return X_train, X_test, y_train, y_test

def binarize(y):
    uy, indices = np.unique(y, return_inverse=True)
    n = y.shape[0]
    y_bin = np.zeros((n, len(uy)))
    for i in range(n):
        y_bin[i, indices[i]] = 1
    return y_bin, uy

# init MNIST kernels by random subimages (patches)
def init_conv(classifier, X_train, n_kernels=None, debug_plot=False):                  
    rnd = np.random.RandomState(0)
    l = classifier.layers[1]  
    wi = l.get_weights()
    img_side = 28
    for i in range(n_kernels):
        f = wi[0][:, :, 0, i]
        w, h = f.shape
        while True:
            i_, j_, k_ = rnd.randint(X_train.shape[0]), rnd.randint(img_side - w), rnd.randint(img_side - h),
            g = np.reshape(X_train[i_, :], (img_side, img_side))[j_:j_ + w, k_:k_ + h]
            g /= np.sum(np.abs(g))
            if np.std(g) > 1e-5:
                break
        wi[0][:, :, 0, i] = g
    classifier.layers[1].set_weights(wi)

    if debug_plot:
        plt.figure()
        l = classifier.layers[1]  
        for i in range(n_kernels):
            f = l.get_weights()[0][:, :, 0, i]
            sqrt_n_kernels = np.ceil(np.sqrt(n_kernels))
            plt.subplot(sqrt_n_kernels, sqrt_n_kernels, i + 1)    
            plt.imshow(f, cmap="gray", interpolation="none")
        plt.show()
    return classifier

def test_keras_random(X_train, X_test, y_train, y_test, n_kernels=225,  debug_plot=True, save_files=False):
    print("TEST KERAS AND RANDOM KERNELS...")  
    
    sqrt_n_kernels = np.ceil(np.sqrt(n_kernels))      
    print("KERNELS: " + str(n_kernels))
    
    path_prefix = "data_out/"
    
    tensorflow.random.set_seed(0)
    np.random.seed(0)
    
    classifier = Sequential()
    classifier.add(Reshape((28, 28, 1), input_shape=(784,)))
    classifier.add(Conv2D(n_kernels, kernel_size=(9, 9)))
    classifier.add(ActivityRegularization(l1=0.000, l2=0.0000001))
    classifier.add(MaxPooling2D(pool_size=(4, 4)))
    classifier.add(Flatten(name="flatten_1"))
    classifier.add(Dense(activation="sigmoid",
                         units=len(np.unique(y_test)), # corresponds to output_dim from previous versions
                         kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                         bias_regularizer=regularizers.l2(1e-4),
                         activity_regularizer=regularizers.l2(1e-5)
                         ))
    
    adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False)
    classifier.compile(optimizer=adam, loss="binary_crossentropy", metrics=["accuracy"])
    by_train = binarize(y_train)[0]
    
    print("KERAS MODEL: ")
    classifier.summary()
    print("FITTING KERAS...")
    t1 = time.time()
    classifier.fit(X_train, by_train, batch_size=50, epochs=100)
    t2 = time.time()
    tkeras = t2 - t1
    print("FITTING KERAS DONE.")
    
    y_pred = classifier.predict(X_test)
    y_pred = np.argmax(y_pred, axis=1)    
    cm = confusion_matrix(y_test, np.unique(y_test)[y_pred])
    print("ACCURACY KERAS: ", accuracy_score(y_test, np.unique(y_test)[y_pred]))
    print("FIT TIME:", tkeras)    
    
    if debug_plot:
        plt.figure()
        for i in range(n_kernels):
            g = classifier.layers[1].get_weights()[0][:, :, 0, i]
            plt.subplot(sqrt_n_kernels, sqrt_n_kernels, i + 1)    
            plt.imshow(g, cmap="gray", interpolation="none")
            plt.axis("off")
        dir = os.getcwd() + "/" + path_prefix
        if not os.path.exists(dir):
            os.makedirs(dir) 
        plt.savefig(path_prefix + "kernels_keras_" + str(n_kernels) + ".eps", bbox_inches="tight")   
    
    print("===============================================================================================================================")
    print("FITTING OTHER CLASSIFIERS ON KERAS KERNELS...")
    t1 = time.time()
    intermediate_layer_model = keras.Model(inputs=classifier.input, outputs=classifier.get_layer("flatten_1").output)
    IX_train = intermediate_layer_model.predict(X_train)
    IX_train = IX_train.astype(np.float64)
    t2 = time.time()    
    print("INTERMEDIATE LAYER PREDICT TIME:", t2 - t1)
    IX_test = intermediate_layer_model.predict(X_test)
    IX_test = IX_test.astype(np.float64)
    
    if save_files:
        print("SAVING DATA...")
        dir = os.getcwd() + "/" + path_prefix
        if not os.path.exists(dir):
            os.makedirs(dir)
        np.savetxt(path_prefix + "mnist_1k_IX_train_keras_" + str(n_kernels) + ".txt", IX_train)
        np.savetxt(path_prefix + "mnist_1k_IX_test_keras_" + str(n_kernels) + ".txt", IX_test)
        np.savetxt(path_prefix + "mnist_1k_X_train" + ".txt", X_train)
        np.savetxt(path_prefix + "mnist_1k_y_train" + ".txt", y_train)        
        np.savetxt(path_prefix + "mnist_1k_X_test" + ".txt", X_test)
        np.savetxt(path_prefix + "mnist_1k_y_test" + ".txt", y_test)
        print("SAVING DATA DONE.")
        
    kernel_ =[]
    time_ = []
    acc_ = []
    clf_ = []
    n_ = []
    m_ = []
    summarg_ = []
    maxmarg_ = []
    time_.append(tkeras)
    acc_.append(np.sum(np.diag(cm)) / np.sum(cm))
    clf_.append("CNN (keras)")
    summarg_.append(0)
    maxmarg_.append(0)
    kernel_.append("keras")
    n_.append(IX_train.shape[1])
    m_.append(IX_train.shape[0])
        
    names = ["cvxopt qp", "libsvm", "liblinear", "liblinear 1e5", "BRP T=1000", "BRP Fast T=1000", "LR + l2", "LR + l1"]
    for clf, name in zip([
                OneVsRestClassifier(SvmCxopt()),
                OneVsRestClassifier(LinearSklearn(clf=SVC(kernel="linear", C=1e7))),
                OneVsRestClassifier(LinearSklearn(clf=LinearSVC(C=1e7))),
                OneVsRestClassifier(LinearSklearn(clf=LinearSVC(C=1e7, max_iter=1e5))), 
                OneVsRestClassifier(BoostedRandomPlanes(T=1000, random_state=0)),
                OneVsRestClassifier(BoostedRandomPlanesFast(T=1000, random_state=0, T0=500, alpha=0.002)),
                OneVsRestClassifier(LinearSklearn(clf=LogisticRegression(C=1e3, penalty="l2", solver="lbfgs"))),
                OneVsRestClassifier(LinearSklearn(clf=LogisticRegression(C=1e1, penalty="l1", solver="liblinear")))], 
                names):
        print("-------------------------------------------------------------------------------------------------------------------------------")
        print("CLASSIFIER: ", name)
        t1 = time.time()
        clf.fit(IX_train, y_train)
        t2 = time.time()
        y1 = clf.predict(IX_test)
        cm = metrics.confusion_matrix(y_test, y1)            
        print("FIT TIME: ", t2 - t1)
        print("CONFUSION MATRIX:")
        print(cm)                        
        mi = [] 
        for j, c in enumerate(clf.estimators_):
            print("MARGIN PER CLASS " + str(j) + ": " + str(c.margin(IX_train, by_train[:, j])[0]))
            mi.append(c.margin(IX_train, by_train[:, j])[0])
        print("ACCURACY: ", metrics.accuracy_score(y_test, y1))    
            
        time_.append(t2 - t1)
        acc_.append(metrics.accuracy_score(y_test, y1))
        clf_.append(name)
        summarg_.append(np.sum(mi))
        maxmarg_.append(np.max(mi))
        kernel_.append("keras")
        n_.append(IX_train.shape[1])
        m_.append(IX_train.shape[0])
      
    print("===============================================================================================================================")
    print("FITTING OTHER CLASSIFIERS ON RANDOM KERNELS...")
    t1 = time.time()
    classifier = init_conv(classifier, X_train, n_kernels=n_kernels)
    intermediate_layer_model = keras.Model(inputs=classifier.input, outputs=classifier.get_layer("flatten_1").output)
    IX_train = intermediate_layer_model.predict(X_train)
    t2 = time.time()
    print("INTERMEDIATE LAYER PREDICT TIME: ", t2 - t1)
    IX_test = intermediate_layer_model.predict(X_test)
    IX_test = IX_test.astype(np.float64)
    
    if save_files:
        np.savetxt(path_prefix + "mnist_1k_IX_train_random_" + str(n_kernels) + ".txt", IX_train)
        np.savetxt(path_prefix + "mnist_1k_IX_test_random_" + str(n_kernels) + ".txt", IX_test)
         
    l = classifier.layers[1]  
    if debug_plot:
        plt.figure()
        for i in range(n_kernels):
            f = l.get_weights()[0][:, :, 0, i]
            plt.subplot(sqrt_n_kernels, sqrt_n_kernels, i + 1)    
            plt.imshow(f, cmap="gray", interpolation="none")
            plt.axis("off") 
        plt.savefig(path_prefix + "kernels_random_" + str(n_kernels) + ".eps", bbox_inches='tight')   
     
    names = ["cvxopt qp", "libsvm", "liblinear", "liblinear 1e5", "BRP T=1000", "BRP Fast T=1000", "LR + l2", "LR + l1"]
    for clf, name in zip([
                OneVsRestClassifier(SvmCxopt()),
                OneVsRestClassifier(LinearSklearn(clf=SVC(kernel="linear", C=1e7))),
                OneVsRestClassifier(LinearSklearn(clf=LinearSVC(C=1e7))),
                OneVsRestClassifier(LinearSklearn(clf=LinearSVC(C=1e7, max_iter=1e5))), 
                OneVsRestClassifier(BoostedRandomPlanes(T=1000, random_state=0)),
                OneVsRestClassifier(BoostedRandomPlanesFast(T=1000, random_state=0, T0=500, alpha=0.002)),
                OneVsRestClassifier(LinearSklearn(clf=LogisticRegression(C=1e3, penalty="l2", solver="lbfgs"))),
                OneVsRestClassifier(LinearSklearn(clf=LogisticRegression(C=1e1, penalty="l1", solver="liblinear")))], 
                names):        
        print("-------------------------------------------------------------------------------------------------------------------------------")
        print("CLASSIFIER: ", name)
        t1 = time.time()
        clf.fit(IX_train, y_train)
        t2 = time.time()
        y1 = clf.predict(IX_test)
        cm = metrics.confusion_matrix(y_test, y1)            
        print("FIT TIME: ", t2 - t1)
        print("CONFUSION MATRIX: ")
        print(cm)
        mi = []
        for j, c in enumerate(clf.estimators_):
            print("MARGIN PER CLASS " + str(j) + ": " + str(c.margin(IX_train, by_train[:, j])[0]))
            mi.append(c.margin(IX_train, by_train[:, j])[0])         
        print("ACCURACY: ", metrics.accuracy_score(y_test, y1))
        
        time_.append(t2 - t1)
        acc_.append(metrics.accuracy_score(y_test, y1))
        clf_.append(name)
        summarg_.append(np.sum(mi))
        kernel_.append("random")
        n_.append(IX_train.shape[1])
        m_.append(IX_train.shape[0])
             
    df  = pd.DataFrame({"kernel" : kernel_, "name" : clf_, "n" : n_, "time": time_, "sum of margins": summarg_, "acc" : acc_ })
    print("TEST KERAS AND RANDOM KERNELS DONE.")
    return df

def test_raw_pixels(X_train, X_test, y_train, y_test):
    print("TEST RAW PIXELS...")  
    time_ = []
    acc_ = []
    clf_ = []
    n_ = []
    m_ = []
    summarg_ = []
    maxmarg_ = []
    minmarg_ = []
    by_train = binarize(y_train)[0]
    
    names=["cvxopt qp", "libsvm", "liblinear", "liblinear 1e5", "BRP T=1000", "BRP Fast T=1000", "LR + l2", "LR + l1"]
    for clf, name in zip([
                OneVsRestClassifier(SvmCxopt()),
                OneVsRestClassifier(LinearSklearn(clf=SVC(kernel="linear", C=1e7))),
                OneVsRestClassifier(LinearSklearn(clf=LinearSVC(C=1e7))),
                OneVsRestClassifier(LinearSklearn(clf=LinearSVC(C=1e7, max_iter=1e5))), 
                OneVsRestClassifier(BoostedRandomPlanes(T=1000, random_state=0)),
                OneVsRestClassifier(BoostedRandomPlanesFast(T=1000, random_state=0, T0=500, alpha=0.002)),
                OneVsRestClassifier(LinearSklearn(clf=LogisticRegression(C=1e3, penalty="l2", solver="lbfgs"))),
                OneVsRestClassifier(LinearSklearn(clf=LogisticRegression(C=1e1, penalty="l1", solver="liblinear")))], 
                names):        
        print("-------------------------------------------------------------------------------------------------------------------------------")
        print("CLASSIFIER: ", name)
        t1 = time.time()
        clf.fit(X_train, y_train)
        t2 = time.time()
        y1 = clf.predict(X_test)
        
        cm = metrics.confusion_matrix(y_test, y1)
        print("FIT TIME: ", t2 - t1)
        print("CONFUSION MATRIX: ")
        print(cm)
        mi = [] 
        for j, c in enumerate(clf.estimators_):        
            print("MARGIN PER CLASS " + str(j) + ": " + str(c.margin(X_train, by_train[:, j])[0]))
            mi.append(c.margin(X_train, by_train[:, j])[0])
        
        print(metrics.accuracy_score(y_test, y1))
        time_.append(t2 - t1)
        acc_.append(metrics.accuracy_score(y_test, y1))
        clf_.append(name)
        summarg_.append(np.sum(mi))
        n_.append(X_train.shape[1])
        m_.append(X_train.shape[0])
        
    df  = pd.DataFrame({"name" : clf_, "n" :  n_, "time" : time_, "sum of margins": summarg_, "acc" : acc_ })
    print("TEST RAW PIXELS DONE.")  
    return df

if __name__=="__main__":
    X_train, X_test, y_train, y_test = load_mnist(train_samples=1000)
    df = None
    
    # --- learning from raw pixels ---
    #df = test_raw_pixels(X_train, X_test, y_train, y_test)    
    
    # --- deep learning and extreme learning ---
    df = test_keras_random(X_train, X_test, y_train, y_test, n_kernels=256, debug_plot=True, save_files=True)
        
    print(df)