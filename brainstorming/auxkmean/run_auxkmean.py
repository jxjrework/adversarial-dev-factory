# examples to show how to use the class
from auxkmean import auxkmean
import time


# Example 1
# initialize an object, only two parameters - size=100 and ncluster=4
aa = auxkmean(100, 5)
# train the object with all files in the path "./trainimgs/" as a parameter
# a csv report will be generated in the same path
# trained model will be saved in pickle format at current path
aa.train_classifier("./trainimgs/")
# predict all files in the path with trained object
# a .csv report will be generated in the same path
aa.predict("./test/")


# Example 2
time.sleep(2)
bb = auxkmean(100, 4)
# without training, object bb can directly import a pre-trained model
# model .pickle file can be appointed as a string parameter here, or
# default model is "auxkmean_model.pickle"
bb.importmodel()
# use imported model to do prediction on all images in folder test2
bb.predict("./test2/")
