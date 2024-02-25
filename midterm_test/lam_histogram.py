import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('male_female_X_train.txt')

classes = np.loadtxt('male_female_Y_train.txt')

heights = data[:, 0]
weights = data[:, 1]
labels = classes[:]

male_heights = heights[labels == 0]
female_heights = heights[labels == 1]

plt.hist(male_heights, bins=20, alpha=0.5, label='Male', color='blue')
plt.hist(female_heights, bins=20, alpha=0.5, label='Female', color='red')

plt.ylabel('Frequency')
plt.xlabel('Heights')
plt.legend()

plt.show()