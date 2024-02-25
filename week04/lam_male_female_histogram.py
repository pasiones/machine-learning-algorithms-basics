import numpy as np
import matplotlib.pyplot as plt

training_data = np.loadtxt('male_female_X_train.txt')
labels = np.loadtxt('male_female_y_train.txt')

heights = training_data[:, 0]
weights = training_data[:, 1]

male_heights = heights[labels == 0]
female_heights = heights[labels == 1]

male_weights = weights[labels == 0]
female_weights = weights[labels == 1]


plt.hist(male_heights, bins = 10, 
         range = (80, 220), alpha = 0.5, label = 'Male', color = 'blue')
plt.hist(female_heights, bins = 10, 
         range = (80, 220), alpha = 0.5, label = 'Female', color = 'red')
plt.xlabel('Heights')
plt.ylabel('Frequency')
plt.legend()
plt.title('Histogram of male and female heights')
plt.show()


plt.hist(male_weights, bins = 10, 
         range = (30, 180), alpha = 0.5, label = 'Male', color = 'blue')
plt.hist(female_weights, bins = 10, 
         range = (30, 180), alpha = 0.5, label = 'Female', color = 'red')
plt.xlabel('Weights')
plt.ylabel('Frequency')
plt.legend()
plt.title('Histogram of male and female weights')
plt.show()