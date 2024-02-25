import numpy as np
import matplotlib.pyplot as plt

tr_data = np.loadtxt('X_train.txt')
tr_classes = np.loadtxt('y_train.txt')
te_data = np.loadtxt('X_test.txt')
te_classes = np.loadtxt('y_test.txt')

tr_heights = tr_data[:, 0]
tr_weights = tr_data[:, 1]

te_heights = te_data[:, 0]
te_weights = te_data[:, 1]

tr_labels = tr_classes[:]
te_labels = te_classes[:]

tr_male_heights = tr_heights[tr_labels == 0]
tr_female_heights = tr_heights[tr_labels == 1]

num_test_samples = len(te_labels)
all_male_prediction = [0] * num_test_samples
all_female_prediction = [1] * num_test_samples

def accuracy_score(predicted_label, test_label = te_labels, num_test_sample = num_test_samples):
    correct_prediction = sum(1 for test, predicted in zip(test_label, predicted_label) 
                             if test == predicted)
    accuracy = correct_prediction / num_test_sample
    return accuracy

all_male_accuracy = accuracy_score(all_male_prediction)
print(f"All male classification accuracy: {all_male_accuracy * 100:.2f}%")

all_female_accuracy = accuracy_score(all_female_prediction)
print(f"All female classification accuracy: {all_female_accuracy * 100:.2f}%")

def gaussian_pdf(x, mu, sigma):
    pdf = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * (((x - mu) / sigma) ** 2))
    return pdf

male_mean = np.mean(tr_male_heights)
male_std = np.std(tr_male_heights)
female_mean = np.mean(tr_female_heights)
female_std = np.std(tr_female_heights)

tr_male_likelihoods = gaussian_pdf(tr_male_heights, male_mean, male_std)
tr_female_likelihoods = gaussian_pdf(tr_female_heights, female_mean, female_std) 

priori_male = len(tr_male_heights) / len(tr_labels)
priori_female = len(tr_female_heights) / len(tr_labels)

te_male_likelihoods = gaussian_pdf(te_heights, male_mean, male_std) 
te_female_likelihoods = gaussian_pdf(te_heights, female_mean, female_std)

P_male_height = priori_male * te_male_likelihoods
P_female_height = priori_female * te_female_likelihoods

predicted_labels = np.zeros(te_labels.shape)
predicted_labels[np.argwhere(np.greater(P_male_height,P_female_height) == True)] = 0
predicted_labels[np.argwhere(np.greater(P_male_height,P_female_height) == False)] = 1
bayes_accuracy = accuracy_score(predicted_labels)
print(f"Bayes classification accuracy: {bayes_accuracy * 100:.2f}%")

plt.scatter(tr_male_heights, tr_male_likelihoods, c='k', label='Male', marker='o')
plt.scatter(tr_female_heights, tr_female_likelihoods, c='r', label='Female', marker='o')
plt.xlabel('Height')
plt.ylabel('Likelihood')
plt.legend()
plt.show()