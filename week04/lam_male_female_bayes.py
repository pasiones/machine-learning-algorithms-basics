import numpy as np
# Load data
test_data = np.loadtxt("male_female_X_test.txt")
test_labels = np.loadtxt("male_female_y_test.txt")

training_data = np.loadtxt('male_female_X_train.txt')
training_labels = np.loadtxt('male_female_y_train.txt')

# Get training data for heights and weights 
training_heights = training_data[:, 0]
training_weights = training_data[:, 1]

training_male_heights = training_heights[training_labels == 0]
training_female_heights = training_heights[training_labels == 1]

training_male_weights = training_weights[training_labels == 0]
training_female_weights = training_weights[training_labels == 1]

# Get test data for heights and weights
test_heights = test_data[:, 0]
test_weights = test_data[:, 1]

# Initialize size of training data, number of males and females
male_num = len(training_male_heights)
female_num = len(training_female_weights)
total_num = len(training_labels)

# Calculation accuracy for classification
def accuracy_calculation(classified_labels, 
                         test_labels = test_labels):
    correct = sum(1 for test, classified 
                  in zip(test_labels, classified_labels) if test == classified)
    accuracy = correct / len(test_labels) * 100
    return accuracy

# Calculate likelihood of test data 
def calculate_likelihood(test_data, hist_gender, hist_all, bin_centroids, p_gender, gender_num , total_num = total_num ):
    abs_differences = [np.abs(i - test_data) for i in bin_centroids]
    bin_index = np.argmin(abs_differences)

    if hist_all[bin_index] != 0:
        likelihood = (hist_gender[bin_index] / gender_num) / (hist_all[bin_index] / total_num) * p_gender
    else:
        likelihood = 0
    return likelihood

# Create histograms for training data
hist_male_heights, _ = np.histogram(training_male_heights, bins = 10, range = (80, 220))
hist_female_heights, _ = np.histogram(training_female_heights, bins = 10, range = (80, 220))
hist_male_weights, _ = np.histogram(training_male_weights, bins = 10, range = (30, 180))
hist_female_weights, _ = np.histogram(training_female_weights, bins = 10, range = (30, 180))
hist_all_heights, bin_edges_height = np.histogram(training_heights, bins = 10, range = (80, 220))
hist_all_weights, bin_edges_weight = np.histogram(training_weights, bins = 10, range = (30, 180))

# A prior of male and female
p_male = len(training_male_heights) / len(training_heights)
p_female = len(training_female_heights) / len(training_heights)

# Calculate bin_centroid
bin_centroids_heights = [(bin_edges_height[i] + bin_edges_height[i + 1]) / 2 
                 for i in range(len(bin_edges_height) - 1)]
bin_centroids_weights = [(bin_edges_weight[i] + bin_edges_weight[i + 1]) / 2 
                 for i in range(len(bin_edges_weight) - 1)]

# Initialize label for classification
classified_label_height = []
classified_label_weight = []
classified_label_height_and_weight = []

# Classify all test data based on height, weight and both 
for sample_height, sample_weight in zip(test_heights, test_weights):
    p_male_h = calculate_likelihood(sample_height, hist_male_heights, 
                                    hist_all_heights, bin_centroids_heights, p_male, male_num)
    p_female_h = calculate_likelihood(sample_height, hist_female_heights, 
                                      hist_all_heights, bin_centroids_heights, p_female, female_num)
    
    p_male_w = calculate_likelihood(sample_weight, hist_male_weights, 
                                    hist_all_weights, bin_centroids_weights, p_male, male_num)
    p_female_w = calculate_likelihood(sample_weight, hist_female_weights, 
                                      hist_all_weights, bin_centroids_weights, p_female, female_num)
    
    if p_male_h > p_female_h:
        classified_label_height.append(0)
    else:
        classified_label_height.append(1)

    if p_male_w > p_female_w:
        classified_label_weight.append(0)
    else:
        classified_label_weight.append(1)

    if p_male_h * p_male_w > p_female_h * p_female_w:
        classified_label_height_and_weight.append(0)
    else:
        classified_label_height_and_weight.append(1)        

# Print out a priori for male and female
print(f"Prior probability for male: {p_male * 100:.2f}%")
print(f"Prior probability for female: {p_female * 100:.2f}%")

# Print out accuracy for classification criteria
print(f"Accuracy for heights only: {accuracy_calculation(classified_label_height):.2f}%")
print(f"Accuracy for weights only: {accuracy_calculation(classified_label_weight):.2f}%")
print(f"Accuracy for both heights and weights: {accuracy_calculation(classified_label_height_and_weight):.2f}%")