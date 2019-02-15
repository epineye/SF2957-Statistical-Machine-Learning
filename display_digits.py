# Display images from the digits data set

import matplotlib.pyplot as plt
from sklearn import datasets

# import some data to play with
digits = datasets.load_digits()

print(digits.data.shape)
plt.gray() 
plt.matshow(digits.images[1]) 
plt.show() 
