#title           :brp_paper_python_experiments.py
#description     :Python script accompanying research paper: "Boosted Random Planes: a Simple Algorithm with Results that Converge in Probability to Hard-Margin SVM Solution" submitted to JMLR. 
#author          :Przemysław Klęsk (pklesk@wi.zut.edu.pl)
#date            :20190819
#version         :1.0
#license         :CC-BY 4.0   
#notes           :The script allows to repeat experiments related to the specified paper for three SVM solvers (libsvm, liblinear, cvxopt). Experiments shall be reproduced exactly using randomization seed equal 0. 
#notes           :BRP solvers (brp, brp_fast, brp_logits) are only optional. To reproduce experiments related to them exactly as in paper, a more efficient Mathematica implementation (compiled to C) should be used instead (notebook: brp.nb).
#acks            :This work was financed by the National Science Centre, Poland. Research project no.: 2016/21/B/ST6/01495.

import argparse
import time
import numpy as np
import csv
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
import cvxopt
from brp_python_implementation import BoostedRandomPlanes, BoostedRandomPlanesFast, BoostedRandomPlanesLogits

def cvxopt_svm(X, y):
    m, n = X.shape
    P = np.zeros((n + 1, n + 1))
    P[1 : n + 1, 1 : n + 1] = np.eye(n)
    q = np.zeros(n + 1)
    y_tiled = np.tile(np.array([y]).T, n + 1)
    G = np.c_[np.ones((m, 1)), X]
    G = -G * y_tiled
    h = -np.ones(m)
    P_cm = cvxopt.matrix(P)
    q_cm = cvxopt.matrix(q)
    G_cm = cvxopt.matrix(G)
    h_cm = cvxopt.matrix(h)
    cvxopt.solvers.options['show_progress'] = False
    return cvxopt.solvers.qp(P_cm, q_cm, G_cm, h_cm)

def data_normals(filepath):
    Xy = np.genfromtxt(filepath, delimiter='\t').astype(float)
    X = Xy[:, :-1]
    y = Xy[:, -1].astype(int)                
    return X, y

def data_iris(filepath):
    data_as_list = []
    with open(filepath) as csv_file:
        read_csv = csv.reader(csv_file, delimiter=',')
        for row in read_csv:
            row[-1] = '-1' if row[-1] == 'Iris-setosa' else '1'
            data_as_list.append(row)
    Xy = np.array(data_as_list).astype(np.float)
    X = Xy[:, :-1]
    y = Xy[:, -1].astype(int)      
    return X, y

def data_sonar(filepath):
    data_as_list = []
    with open(filepath) as csv_file:
        read_csv = csv.reader(csv_file, delimiter=',')
        for row in read_csv:
            row[-1] = '-1' if row[-1] == 'R' else '1'
            data_as_list.append(row)
    Xy = np.array(data_as_list).astype(np.float)
    X = Xy[:, :-1]
    y = Xy[:, -1].astype(int)      
    return X, y

def data_leukemia(filepath):
    data_as_list = []
    row_y = []    
    with open(filepath) as csv_file:
        read_csv = csv.reader(csv_file, delimiter=',')
        for i, row in enumerate(read_csv):
            if i == 0:
                row_y = row
            else:            
                data_as_list.append(row)
    for i in range(len(row_y)):
        row_y[i] = '-1' if row_y[i] == 'ALL' else '1'
    data_as_list.append(row_y)
    Xy = np.array(data_as_list).astype(np.float)
    Xy = Xy.T
    X = Xy[:, :-1]
    y = Xy[:, -1].astype(int)      
    return X, y

def data_ionosphere(filepath):
    data_as_list = []
    with open(filepath) as csv_file:
        read_csv = csv.reader(csv_file, delimiter=',')
        for row in read_csv:
            row[-1] = '1' if row[-1] == 'g' else '-1'
            data_as_list.append(row)
    Xy = np.array(data_as_list).astype(np.float)
    X = Xy[:, :-1]
    y = Xy[:, -1].astype(int)      
    return X, y

def data_spambase(filepath):
    data_as_list = []
    with open(filepath) as csv_file:
        read_csv = csv.reader(csv_file, delimiter=',')
        for row in read_csv:
            row[-1] = '1' if row[-1] == '0' else '-1'
            data_as_list.append(row)
    Xy = np.array(data_as_list).astype(np.float)
    X = Xy[:, :-1]
    y = Xy[:, -1].astype(int)      
    return X, y

def data_processed(filepath):
    Xy = np.genfromtxt(filepath, delimiter='\t').astype(float)
    X = Xy[:, :-1]
    y = Xy[:, -1].astype(int)       
    return X, y

if __name__ == "__main__":
    path = "./data/"
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=str, choices=["normals2d_small", "normals3d_small", "normals2d_10k", "normals3d_10k", "normals2d_100k", "normals3d_100k", "iris", "sonar", "leukemia", "ionosphere", "spambase"], help="specify data")
    parser.add_argument("solver", type=str, choices=["libsvm", "liblinear", "cvxopt", "brp", "brp_fast", "brp_logits"], help="specify solver")     
    parser.add_argument("-repetitions", type=int, default=1, help="specify number of repetitions (to estimate average time per execution, default: 1)")
    parser.add_argument("-C", type=float, default=1e7, help="specify value for C parameter (only for libsvm or liblinear, default: 1e7)")
    parser.add_argument("-max_iter", type=float, default=1e3, help="specify maximum number of iterations to be run (only for liblinear, default: 1e3)")
    parser.add_argument("-kernel", type=str, choices=["linear", "rbf", "poly"], default="linear", help="specify type of kernel (only for libsvm, default: linear)")
    parser.add_argument("-T", type=int, default=1000, help="specify number of boosting rounds (only for brp or brp_fast or brp_logits, default: 1000)")
    parser.add_argument("-T0", type=int, default=100, help="specify number of initial boosting rounds (only for brp_fast, default: 100)")
    parser.add_argument("-alpha", type=float, default=0.05, help="specify trimming ratio (only for brp_fast, default: 0.05)")
    parser.add_argument("-random_state", type=int, default=0, help="specify randomization seed (default: 0)")    
    args = parser.parse_args()
    
    X_train = None
    y_train = None
    X_test = None
    y_test = None
    if args.data == "normals2d_small":        
        X_train, y_train = data_normals(path + "normals2d_small.txt")
    elif args.data == "normals3d_small":
        X_train, y_train = data_normals(path + "normals3d_small.txt")
    elif args.data == "normals2d_10k":
        X_train, y_train = data_normals(path + "normals2d_10k.txt")
    elif args.data == "normals3d_10k":
        X_train, y_train = data_normals(path + "normals3d_10k.txt")
    elif args.data == "normals2d_100k":
        X_train, y_train = data_normals(path + "normals2d_100k.txt")
    elif args.data == "normals3d_100k":
        X_train, y_train = data_normals(path + "normals3d_100k.txt")                
    elif args.data == "iris":
        X_train, y_train = data_iris(path + "iris.txt")
    elif args.data == "sonar":
        X_train, y_train = data_sonar(path + "sonar.all-data")
    elif args.data == "leukemia":
        X_train, y_train = data_leukemia(path + "leukemia_big.csv")        
    elif args.data == "ionosphere":
        X, y = data_ionosphere(path + "ionosphere.data")
        X_train, y_train = data_processed(path + "ionosphere_train.txt")
        X_test, y_test= data_processed(path + "ionosphere_test.txt")
    elif args.data == "spambase":            
        X, y = data_spambase(path + "spambase.data")
        X_train, y_train = data_processed(path + "spambase_train.txt")
        X_test, y_test= data_processed(path + "spambase_test.txt")
    m, n = X_train.shape[0], X_train.shape[1]
    print("DATA: " + args.data + " [" + str(m) + " x " + str(n) + "].")               

    print("SOLVER: " + args.solver + ".")
    print("REPETITIONS: " + str(args.repetitions) + ".")
    print("RUNNING...")    
    if args.solver == "libsvm":
        clf = SVC(kernel=args.kernel, C=args.C, random_state=args.random_state)    
        total_time = 0.0     
        for t in range(args.repetitions):
            t1 = time.time()            
            clf.fit(X_train, y_train)
            t2 = time.time()
            total_time += t2 - t1
        total_time /= args.repetitions
        print("DONE.")
        print("TIME PER EXECUTION: " + str(1000 * total_time) + " [ms].")
        print("SUPPORT VECTORS (NEG., POS.): (" + str(clf.n_support_[0]) + ", " + str(clf.n_support_[1]) + ").")        
        if args.kernel == "linear":
            clf_v = np.c_[np.array([clf.intercept_]), clf.coef_].ravel()
            print("MARGIN: " + str(np.min(y_train * (np.c_[np.ones((X_train.shape[0], 1)), X_train]).dot(clf_v.T) / np.linalg.norm(clf.coef_))) + ".")        
        print("TRAIN ACCURACY: " + str(clf.score(X_train, y_train)) + ".")
        if X_test is not None:
            print("TEST ACCURACY: " + str(clf.score(X_test, y_test)) + ".")                        
    elif args.solver == "liblinear":
        clf = LinearSVC(C=args.C, max_iter=args.max_iter, random_state=args.random_state)
        total_time = 0.0            
        for t in range(args.repetitions):
            t1 = time.time()            
            clf.fit(X_train, y_train)
            t2 = time.time()
            total_time += t2 - t1
        total_time /= args.repetitions
        print("DONE.")
        print("TIME PER EXECUTION: " + str(1000 * total_time) + " [ms].")        
        clf_v = np.c_[np.array([clf.intercept_]), clf.coef_].ravel()
        print("MARGIN: " + str(np.min(y_train * (np.c_[np.ones((X_train.shape[0], 1)), X_train]).dot(clf_v.T) / np.linalg.norm(clf.coef_))) + ".")
        print("TRAIN ACCURACY: " + str(clf.score(X_train, y_train)) + ".")
        if X_test is not None:
            print("TEST ACCURACY: " + str(clf.score(X_test, y_test)) + ".")    
    elif args.solver == "cvxopt":
        total_time = 0.0
        for t in range(args.repetitions):
            t1 = time.time()            
            cvxopt_solution = cvxopt_svm(X_train, y_train)
            t2 = time.time()
            total_time += t2 - t1
        total_time /= args.repetitions
        print("DONE.")
        print("TIME PER EXECUTION: " + str(1000 * total_time) + " [ms].")                
        solution = np.array(cvxopt_solution['x'])
        supp = np.where(np.array(cvxopt_solution['z']) > 1e-5)[0]        
        print("SUPPORT VECTORS (NEG., POS.): (" + str(np.where(y_train[supp] == -1)[0].size) + ', ' + str(np.where(y_train[supp] == 1)[0].size) + ').')        
        v0 = np.array([[solution[0, 0]]])
        v = np.array([solution[1 : n + 1, 0]])        
        print("MARGIN: " + str(1.0 / np.linalg.norm(v)) + ".")
    elif args.solver == "brp":
        clf = BoostedRandomPlanes(T=args.T, random_state=args.random_state)
        total_time = 0.0            
        for t in range(args.repetitions):
            t1 = time.time()            
            clf.fit(X_train, y_train)
            t2 = time.time()
            total_time += t2 - t1
        total_time /= args.repetitions
        print("DONE.")
        print("TIME PER EXECUTION: " + str(1000 * total_time) + " [ms].")        
        clf_v = np.concatenate((np.array([clf.intercept_]), clf.coef_))
        print("MARGIN: " + str(np.min(y_train * (np.c_[np.ones((X_train.shape[0], 1)), X_train]).dot(clf_v.T) / np.linalg.norm(clf.coef_))) + ".")
        print("TRAIN ACCURACY: " + str(clf.score(X_train, y_train)) + ".")
        if X_test is not None:
            print("TEST ACCURACY: " + str(clf.score(X_test, y_test)) + ".")    
    elif args.solver == "brp_fast":
        clf = BoostedRandomPlanesFast(T=args.T, T0=args.T0, alpha=args.alpha, random_state=args.random_state)
        total_time = 0.0            
        for t in range(args.repetitions):
            t1 = time.time()            
            clf.fit(X_train, y_train)
            t2 = time.time()
            total_time += t2 - t1
        total_time /= args.repetitions
        print("DONE.")
        print("TIME PER EXECUTION: " + str(1000 * total_time) + " [ms].")        
        clf_v = np.concatenate((np.array([clf.intercept_]), clf.coef_))
        print("MARGIN: " + str(np.min(y_train * (np.c_[np.ones((X_train.shape[0], 1)), X_train]).dot(clf_v.T) / np.linalg.norm(clf.coef_))) + ".")
        print("TRAIN ACCURACY: " + str(clf.score(X_train, y_train)) + ".")
        if X_test is not None:
            print("TEST ACCURACY: " + str(clf.score(X_test, y_test)) + ".")
    elif args.solver == "brp_logits":
        clf = BoostedRandomPlanesLogits(T=args.T, random_state=args.random_state)
        total_time = 0.0            
        for t in range(args.repetitions):
            t1 = time.time()            
            clf.fit(X_train, y_train)
            t2 = time.time()
            total_time += t2 - t1
        total_time /= args.repetitions
        print("DONE.")
        print("TIME PER EXECUTION: " + str(1000 * total_time) + " [ms].")
        print("TRAIN ACCURACY: " + str(clf.score(X_train, y_train)) + ".")
        if X_test is not None:
            print("TEST ACCURACY: " + str(clf.score(X_test, y_test)) + ".")
    print("")