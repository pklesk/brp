# title           :brp_python_implementation.py
# description     :Example Python implementation of Boosted Random Planes algorithm (and variants), accompanying research paper: "Can Boosted Randomness Mimic Learning Algorithms of Geometric Nature? Example of a Simple Algorithm that Converges in Probability to Hard-Margin SVM". 
# author          :Przemysław Klęsk, Marcin Korzeń
# date            :20200827
# version         :1.1
# license         :CC-BY 4.0
# acks            :This work was financed by the National Science Centre, Poland. Research project no.: 2016/21/B/ST6/01495.

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.svm import LinearSVC
import numpy as np
import time

import cvxopt

class SvmBase(BaseEstimator, ClassifierMixin):
    
    def __init__(self,):
        self.coef_ = None
        self.intercept_ = None    
        
    def margin(self, X, y):
        y = 2 * (y > 0) - 1
        coefs = np.concatenate((np.array([self.intercept_]), self.coef_))
        margs = (y * (np.c_[np.ones((X.shape[0], 1)), X]).dot(coefs)) / np.linalg.norm(self.coef_)
        return np.min(margs), margs

    def decision_function(self, X):
        tau, margs = self.margin(X, np.ones((X.shape[0])))
        return margs
    
    def predict_proba(self, X):
        tau, margs = self.margin(X, np.ones((X.shape[0])))
        return np.array([1.0 - 1.0 / (1.0 + np.exp(-margs)), 1.0 / (1.0 + np.exp(-margs))]).T

    def predict(self, X):
        coef_all = np.concatenate((np.array([self.intercept_]), self.coef_))
        X_ext = np.c_[np.ones((X.shape[0], 1)), X]
        results = np.sign(X_ext.dot(coef_all.T))        
        results_mapped = self.class_labels_[1 * (results > 0)]
        return results_mapped

  
class LinearSklearn(SvmBase):

    def __init__(self, clf=LinearSVC()):
        super(LinearSklearn, self).__init__()
        self.clf = clf

    def fit(self, X, y):
        self.class_labels_ = np.unique(y) 
        self.clf.fit(X, y)
        self.coef_ = self.clf.coef_[0]
        self.intercept_ = self.clf.intercept_[0]
        
class SvmCxopt(SvmBase):
    
    def __init__(self,):
        super(SvmCxopt, self).__init__()
        
    def fit(self, X, y):
        y = 2 * (y > 0) - 1
        self.class_labels_ = np.unique(y) 
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
        self.solution = cvxopt.solvers.qp(P_cm, q_cm, G_cm, h_cm)
        self.coef_ = np.array(self.solution['x'][1:, 0]).ravel()
        self.intercept_ = self.solution['x'][0]
       

class BoostedRandomPlanes(SvmBase):
    
    def __init__(self, T=1000, random_state=0):
        self.T_ = T
        self.random_state_ = random_state
        self.coef_ = None
        self.intercept_ = None    
        self.class_labels_ = None  
        
    def fit(self, X, y):                
        m, n = X.shape
        self.class_labels_ = np.unique(y)                
        rnd = np.random.RandomState(self.random_state_)                
        ind_neg = np.where(y == self.class_labels_[0])[0]  # assuming first unique label is negative class
        ind_pos = np.where(y == self.class_labels_[1])[0]  # assuming first unique label is negative class
        X_neg = X[ind_neg]
        X_pos = X[ind_pos]
        w_neg = np.ones(ind_neg.size) / (1.0 * ind_neg.size)
        w_pos = np.ones(ind_pos.size) / (1.0 * ind_pos.size)
        V = np.zeros(n)
        P = np.zeros(n)                        
        for t in range(self.T_):              
            i = ind_neg[rnd.choice(ind_neg.size, p=w_neg)]
            j = ind_pos[rnd.choice(ind_pos.size, p=w_pos)]            
            v = X[j] - X[i]
            v /= np.linalg.norm(v)
            p = 0.5 * (X[i] + X[j]) 
            V += v
            P += p            
            w_neg = w_neg * np.exp(X_neg.dot(v.T))            
            w_pos = w_pos * np.exp(-X_pos.dot(v.T))
            w_neg /= w_neg.sum()
            w_pos /= w_pos.sum()                
        self.coef_ = V / np.linalg.norm(V)
        P /= self.T_
        self.intercept_ = -self.coef_.dot(P.T)        
         
    def predict(self, X):
        coef_all = np.concatenate((np.array([self.intercept_]), self.coef_))
        X_ext = np.c_[np.ones((X.shape[0], 1)), X]
        results = np.sign(X_ext.dot(coef_all.T))        
        results_mapped = np.array(list(map(lambda r: self.class_labels_[0] if r <= 0.0 else self.class_labels_[1], results)))  # mapping {<= 0.0, > 0.0} to first two class labels, respectively
        return results_mapped
    
    def get_params(self, deep=True):
        return {"T": self.T_, "random_state": self.random_state_}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    
class BoostedRandomPlanesFast(SvmBase):
    
    def __init__(self, T=1000, random_state=0, T0=100, alpha=0.05):
        self.T_ = T
        self.random_state_ = random_state
        self.T0_ = T0
        self.alpha_ = alpha
        self.coef_ = None
        self.intercept_ = None    
        self.class_labels_ = None  
        
    def fit(self, X, y):
        m, n = X.shape
        self.class_labels_ = np.unique(y)                
        rnd = np.random.RandomState(self.random_state_)                             
        ind_neg = np.where(y == self.class_labels_[0])[0]  # assuming first unique label is negative class
        ind_pos = np.where(y == self.class_labels_[1])[0]  # assuming first unique label is negative class
        ind_rev = np.zeros(y.size).astype(int)
        ind_rev[ind_neg] = np.arange(ind_neg.size)
        ind_rev[ind_pos] = np.arange(ind_pos.size)
        X_neg = X[ind_neg]
        X_pos = X[ind_pos]
        w_neg = np.ones(ind_neg.size) / (1.0 * ind_neg.size)
        w_pos = np.ones(ind_pos.size) / (1.0 * ind_pos.size)
        events_neg = np.zeros(ind_neg.size)
        events_pos = np.zeros(ind_pos.size)
        V = np.zeros(n)
        P = np.zeros(n)
        T00 = min(self.T_, self.T0_)                        
        for t in range(T00):                               
            i = ind_neg[rnd.choice(ind_neg.size, p=w_neg)]
            j = ind_pos[rnd.choice(ind_pos.size, p=w_pos)]  
            events_neg[ind_rev[i]] += 1
            events_pos[ind_rev[j]] += 1          
            v = X[j] - X[i]
            v /= np.linalg.norm(v)
            p = 0.5 * (X[i] + X[j]) 
            V += v
            P += p            
            w_neg = w_neg * np.exp(X_neg.dot(v.T))
            w_pos = w_pos * np.exp(-X_pos.dot(v.T))        
            w_neg /= w_neg.sum()
            w_pos /= w_pos.sum()   
        if self.T0_ < self.T_:
            trimming_min_events = np.ceil(self.alpha_ * self.T0_)
            w_neg_supp = np.where(np.sign(events_neg - trimming_min_events) == 1.0)[0]
            w_pos_supp = np.where(np.sign(events_pos - trimming_min_events) == 1.0)[0]
            if w_neg_supp.size == 0:
                w_neg_supp = np.arange(ind_neg.size)  # trimming ratio (alpha) too large for negative class, taking all indexes
                print("[TRIMMING RATIO TOO LARGE FOR NEGATIVE CLASS, TAKING ALL INDEXES.]")
            if w_pos_supp.size == 0:
                w_pos_supp = np.arange(ind_pos.size)  # trimming ratio (alpha) too large for positive class, taking all indexes
                print("[TRIMMING RATIO TOO LARGE FOR POSITIVE CLASS, TAKING ALL INDEXES.]")                
            trimmed_indexes = np.concatenate((ind_neg[w_neg_supp], ind_pos[w_pos_supp]))
            X_t = X[trimmed_indexes]
            y_t = y[trimmed_indexes]
            ind_neg_t = np.where(y_t == self.class_labels_[0])[0]
            ind_pos_t = np.where(y_t == self.class_labels_[1])[0]
            X_neg_t = X_t[ind_neg_t]
            X_pos_t = X_t[ind_pos_t]
            w_neg_t = w_neg[w_neg_supp]
            w_pos_t = w_pos[w_pos_supp]
            w_neg_t /= w_neg_t.sum()
            w_pos_t /= w_pos_t.sum()
            print("[TRIMMING DOWN TO (" + str(ind_neg_t.size) + ", " + str(ind_pos_t.size) + ") SUPPORT VECTORS.]")
            for t in range(self.T0_, self.T_ + 1):
                i = ind_neg_t[rnd.choice(ind_neg_t.size, p=w_neg_t)]
                j = ind_pos_t[rnd.choice(ind_pos_t.size, p=w_pos_t)]            
                v = X_t[j] - X_t[i]
                v /= np.linalg.norm(v)
                p = 0.5 * (X_t[i] + X_t[j]) 
                V += v
                P += p            
                w_neg_t = w_neg_t * np.exp(X_neg_t.dot(v.T))
                w_pos_t = w_pos_t * np.exp(-X_pos_t.dot(v.T))
                w_neg_t /= w_neg_t.sum()
                w_pos_t /= w_pos_t.sum()                   
        self.coef_ = V / np.linalg.norm(V)
        P /= self.T_
        self.intercept_ = -self.coef_.dot(P.T)        
         
    def predict(self, X):
        coef_all = np.concatenate((np.array([self.intercept_]), self.coef_))
        X_ext = np.c_[np.ones((X.shape[0], 1)), X]
        results = np.sign(X_ext.dot(coef_all.T))        
        results_mapped = np.array(list(map(lambda r: self.class_labels_[0] if r <= 0.0 else self.class_labels_[1], results)))  # mapping {<= 0.0, > 0.0} to first to class labels, respectively
        return results_mapped

    def get_params(self, deep=True):
        return {"T": self.T_, "T0": self.T0_, "alpha": self.alpha_, "random_state": self.random_state_}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    
class BoostedRandomPlanesLogits(BaseEstimator, ClassifierMixin):
    
    def __init__(self, T=100, random_state=0):
        self.T_ = T
        self.random_state_ = random_state
        self.F_ = None  # ensemble  
        self.class_labels_ = None  
        
    def fit(self, X, y):
        MAX_LOGIT = 2.0
        rnd = np.random.RandomState(self.random_state_)         
        m, n = X.shape
        self.class_labels_ = np.unique(y)                                                
        ind_neg = np.where(y == self.class_labels_[0])[0]  # assuming first unique label is negative class
        ind_pos = np.where(y == self.class_labels_[1])[0]  # assuming first unique label is negative class
        X_neg = X[ind_neg]
        X_pos = X[ind_pos]            
        w_neg = np.ones(ind_neg.size) / (1.0 * ind_neg.size)
        w_pos = np.ones(ind_pos.size) / (1.0 * ind_pos.size)
        self.F_ = np.zeros((self.T_, 1 + n + 2))                        
        for t in range(self.T_):                                
            i = ind_neg[rnd.choice(ind_neg.size, p=w_neg)]
            j = ind_pos[rnd.choice(ind_pos.size, p=w_pos)]            
            v = X[j] - X[i]
            v /= np.linalg.norm(v)
            p = 0.5 * (X[i] + X[j]) 
            v0 = -v.dot(p)
            self.F_[t, 0] = v0
            self.F_[t, 1 : n + 1] = v
            ind_neg_r = np.where(np.array(list(map(lambda x: np.sign(v0 + v.dot(x)), X_neg))) == 1)[0]
            ind_neg_l = np.setdiff1d(np.arange(ind_neg.size), ind_neg_r)
            ind_pos_r = np.where(np.array(list(map(lambda x: np.sign(v0 + v.dot(x)), X_pos))) == 1)[0]
            ind_pos_l = np.setdiff1d(np.arange(ind_pos.size), ind_pos_r)
            logit_l = 0.0
            if ind_neg_l.size == 0 and ind_pos_l.size > 0:
                logit_l = MAX_LOGIT
            elif ind_neg_l.size > 0 and ind_pos_l.size == 0:
                logit_l = -MAX_LOGIT
            elif ind_neg_l.size > 0 and ind_pos_l.size > 0:
                logit_l = 0.5 * np.log(w_pos[ind_pos_l].sum() / w_neg[ind_neg_l].sum())
                if np.abs(logit_l) > MAX_LOGIT:
                    logit_l = np.sign(logit_l) * MAX_LOGIT
            logit_r = 0.0
            if ind_neg_r.size == 0 and ind_pos_r.size > 0:
                logit_r = MAX_LOGIT
            elif ind_neg_r.size > 0 and ind_pos_r.size == 0:
                logit_r = -MAX_LOGIT
            elif ind_neg_r.size > 0 and ind_pos_r.size > 0:
                logit_r = 0.5 * np.log(w_pos[ind_pos_r].sum() / w_neg[ind_neg_r].sum())
                if np.abs(logit_r) > MAX_LOGIT:
                    logit_r = np.sign(logit_r) * MAX_LOGIT
            self.F_[t, 1 + n] = logit_l
            self.F_[t, 1 + n + 1] = logit_r
            if ind_neg_l.size > 0:
                w_neg[ind_neg_l] = w_neg[ind_neg_l] * np.exp(logit_l)
            if ind_neg_r.size > 0:
                w_neg[ind_neg_r] = w_neg[ind_neg_r] * np.exp(logit_r)
            w_neg /= w_neg.sum()                 
            if ind_pos_l.size > 0:
                w_pos[ind_pos_l] = w_pos[ind_pos_l] * np.exp(-logit_l)
            if ind_pos_r.size > 0:
                w_pos[ind_pos_r] = w_pos[ind_pos_r] * np.exp(-logit_r)
            w_pos /= w_pos.sum()                                        
         
    def predict(self, X):
        m, n = X.shape
        X_ext = np.c_[np.ones((X.shape[0], 1)), X]
        results = np.sign(X_ext.dot(self.F_[:, :n + 1].T))
        for i in range(m):
            for t in range(self.T_):
                results[i, t] = self.F_[t, 1 + n] if results[i, t] <= 0.0 else self.F_[t, 1 + n + 1]
        results_mapped_1 = results.sum(axis=1) 
        results_mapped_2 = np.array(list(map(lambda r: self.class_labels_[0] if r <= 0.0 else self.class_labels_[1], results_mapped_1)))  # mapping {<= 0.0, > 0.0} to first to class labels, respectively
        return results_mapped_2