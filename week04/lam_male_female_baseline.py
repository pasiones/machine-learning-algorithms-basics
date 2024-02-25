import random
import numpy as np

test_data = np.loadtxt("male_female_X_test.txt")
test_labels = np.loadtxt("male_female_y_test.txt")

training_data = np.loadtxt('male_female_X_train.txt')
training_labels = np.loadtxt('male_female_y_train.txt')
# Randon classifier
def random_classifier(num_samples):
    return [random.randint(0, 1) for _ in range(num_samples)]

num_test_samples = len(test_labels)

predicted_labels = random_classifier(num_test_samples)

correct_predictions = sum(1 for test, predicted 
                          in zip(test_labels, predicted_labels) if test == predicted)
random_accuracy = correct_predictions / num_test_samples

print(f"Random Classification Accuracy: {random_accuracy * 100:.2f}%")

# Classifier for most likely class
num_training_samples = len(test_labels)
num_male = np.sum(test_labels == 0)
num_female = np.sum(test_labels == 1)

p_male = num_male / num_training_samples
p_female = num_female / num_training_samples

if p_male > p_female:
    classified_label = 0
else:
    classified_label = 1

classified_labels = [classified_label for _ in range(num_test_samples)]

correct = sum(1 for test, classified 
              in zip(test_labels, classified_labels) if test == classified)
most_likely_class_accuracy = correct / num_test_samples

print(f"Highest a priori Classification Accuracy: {most_likely_class_accuracy * 100:.2f}%")

