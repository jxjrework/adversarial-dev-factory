import scipy.io
import numpy as np
from matplotlib import use
use('TkAgg')
from displayData import displayData

def display100(file):
    """Randomly pick 100 images from a file and displays them in a nice grid."""
    # Load Training Data
    print('Loading and Visualizing Data ...')
    data = scipy.io.loadmat(file)
    # training data stored in arrays X, y
    X = data['X']
    y = data['y']
    np.savetxt("y.csv", y)
    m, _ = X.shape
    print(y.shape)
    np.savetxt("newy.csv", y)

    # Randomly select 100 data points to display
    rand_indices = np.random.permutation(range(m))
    sel = X[rand_indices[0:100], :]
    displayData(sel)
