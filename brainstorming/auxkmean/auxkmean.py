from sklearn.cluster import KMeans
import numpy as np
import pickle
from PIL import Image
#from pylab import *
import os, sys
import pandas as pd



class auxkmean(object):


    def __init__(self, size, ncluster):

        # path - directory folder of training dataset
        # size - side length of square RGB images
        # ncluster - number of clusters to group the training set

        self.size = (size, size)
        self.totfeature = size * size * 3
        self.ncluster = ncluster

        # classifier training result
        self.training_outputnames = []
        self.training_outputclass = []

        # classifier predicting result
        self.predict_outputnames = []
        self.predict_outputclass = []
        self.predictmodel = None


    def image2vector(self, image):
        # return a 1-d list of pixel features of the image cropped by self.size
        # image - string of the image file name "/.../xxx.png" including path
        # size - size of the shrinked image format (size, size)
        # transform .png image into a 1-D vector with format 1 x dim
        imobj = Image.open(image).resize(self.size)
        # transform image into an array
        im_array = np.array(imobj)
        im_1d = im_array.ravel()
        return list(im_1d)


    def prepare_dataset(self, datapath):
        # return all filenames as a list and one nparray of all data
        # generate data set as a numpy array
        # each column is one feature
        # each row is one sample imag
        X = np.empty(shape = [1, self.totfeature])
        imagenames = os.listdir(datapath)
        outputnames = []
        for fname in imagenames:
            if fname[-4:]==".png":
                newvector = np.array([self.image2vector(os.path.join(datapath+fname))])
                X = np.append(X, newvector, axis = 0)
                outputnames.append(fname[:-4])
        return outputnames, np.delete(X, (0), axis=0)


    def train_classifier(self, trainpath, outputmodel="auxkmean_model.pickle"):
        # train classifier with images in self.path folder
        # save trained model as outputmodel pickle file
        self.training_outputnames, X = self.prepare_dataset(trainpath)
        kmeans = KMeans(n_clusters=self.ncluster, random_state=0).fit(X)
        #print(kmeans.cluster_centers_)
        #print(kmeans.labels_)
        self.training_outputclass = kmeans.labels_
        self.composecsv(trainpath,
            "TrainingResult.csv",
            self.training_outputnames,
            self.training_outputclass)

        if self.predictmodel!=None:
            print("Replacing current model with newly-trained model...")
        self.predictmodel = kmeans
        try:
            pickle.dump(kmeans, open(outputmodel, 'wb'))
        except:
            print("Failed to save model as file!")
            return
        print("Model successfully trained and saved to file!")


    def importmodel(self, modelname="auxkmean_model.pickle"):
        # in case need to apply pre-saved model without retraining

        if (self.predictmodel != None):
            print("Replacing current model with imported model...")
        try:
            self.predictmodel = pickle.load(open(modelname, 'rb'))
        except:
            print("Failed to import models from file!")
            return
        print("Model successfully imported from file!")


    def predict(self, testpath, ncol=1):
        if (self.predictmodel==None):
            print("Model not ready, please train or load a model!")
            return
        self.predict_outputnames, X = self.prepare_dataset(testpath)
        self.predict_outputclass = list(self.predictmodel.predict(X))
        self.composecsv(testpath, "PredictResult.csv", self.predict_outputnames, self.predict_outputclass)

        # sorted class ranking for each test sample
        if ncol > 1:
            class_ranking = np.argsort(self.predictmodel.transform(X), axis=1)
            if ncol > self.ncluster:
                print("Classranking ncol must be fewer than clusters")
                return
            class_ranking = class_ranking[:, 0:ncol]
            self.composecsv_classranking(testpath, "PredictClassRanking.csv",
                self.predict_outputnames, class_ranking)


    def composecsv_classranking(self, path, reportname, names, class_ranking):
        # args:
        # class_ranking - np matrix ranking the distances from each cluster center, small to large
        # each row in class_ranking matrix is one test sample
        if len(names) == class_ranking.shape[0]:
            colnames = []
            for i in range(class_ranking.shape[1]):
                colnames.append('Category_'+str(i))

            df = pd.DataFrame(data = class_ranking,
                              columns = colnames)
            df.insert(0, 'ImageName', names)
            df.to_csv(os.path.join(path, reportname))
        else:
            print("Class ranking matrix dimension mismatch found!")


    def composecsv(self, path, reportname, names, labels):
        # path - path to save the generated csv report
        # reportname - filename of the report
        # names - names of all sample files, as a list
        # label - class labels for all sample files, as a list

        df = pd.DataFrame()
        se = pd.Series(names)
        df['ImageName'] = se.values
        sd = pd.Series(labels)
        df['kmeanClass'] = sd.values
        df.to_csv(os.path.join(path, reportname))
