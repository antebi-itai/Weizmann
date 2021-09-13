import numpy as np


data = np.load('Drinking Fountain Somewhere In Zurich.npz')
P = data['P']
X = data['X']
x = data['x']
visible_points = data['visible_points']
K = data['K']
