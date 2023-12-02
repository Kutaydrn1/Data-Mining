

print("1. Mean, median, and mode")

import statistics
numbers = [10, 20, 30, 40, 40, 50, 60, 70, 80, 90]

mean = statistics.mean (numbers)
median = statistics.median (numbers)
mode = statistics.mode (numbers)

print("Mean:", mean)
print("Median:", median)
print("Mode:", mode)

print("-"*75)

print("2. Boxplot Analysis")

import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

iris = load_iris()

X = iris.data
y = iris.target

plt.boxplot(X)

plt.title("Box plot - Iris Dataset")
plt.xlabel("Features")
plt.ylabel("Values")
plt.show()

print("-"*75)


print("3. Correlated and Uncorrelated Data")

import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as pt

# correlated data
np.random. seed (0)
correlated_data = pd.DataFrame(np.random.randn(100, 2), columns=['Variable1', 'Variable2'])
correlated_data['Variable2'] = correlated_data['Variable1'] + np.random.randn(100) * 0.5

# uncorrelated data
np.random. seed (1)
uncorrelated_data = pd.DataFrame(np.random. rand (100, 2), columns=['Variable3', 'Variable4'])

# Calculate correlation coefficients
correlation_corr = correlated_data ['Variable1'].corr(correlated_data ['Variable2'])
correlation_uncorr = uncorrelated_data['Variable3'].corr(uncorrelated_data['Variable4'])

print("Correlation coefficient (correlated data):", correlation_corr)
print("Correlation coefficient (uncorrelated data):", correlation_uncorr)

# Visualize
sns.scatterplot(data=correlated_data, x= 'Variable1', y= 'Variable2')
sns.scatterplot(data=uncorrelated_data, x='Variable3' , y= 'Variable4')
plt.show()

print("-"*75)

print("4. Similarity-Dissimilarity")

import numpy as np
from scipy.spatial.distance import cosine

# similarity
def cosineSimilarity(vector1, vector2):
    # computing dot product of the two vectors
    dot_product = np.dot(vector1, vector2)
    # computing the magnitudes of the vectors
    magnitude_1 = np.linalg.norm(vector1)
    magnitude_2 = np.linalg.norm(vector2)
    # computing the cosine similarity
    similarity = dot_product / (magnitude_1 * magnitude_2)
    return similarity

# dissimilarity
def cosineDissimilarity(vector1, vector2):
    # computing cosine similarity
    similarity = cosineSimilarity(vector1, vector2)
    # computing cosine dissimilarity
    dissimilarity = 1 - similarity
    return dissimilarity

vector_1 = np.array([1, 2, 3])
vector_2 = np.array([4, 5, 6])

similarity = cosineSimilarity(vector_1, vector_2)
dissimilarity = cosineDissimilarity(vector_1, vector_2)

print("Cosine Similarity:", similarity)
print("Cosine Dissimilarity:", dissimilarity)

print("-"*75)


print("5. Min Max Normalization")

import numpy as np

# min-max normalization
def min_max_normalize(data):
    # minimum and maximum values in the data
    minValue = np.min(data)
    maxValue = np.max(data)

    # normalizing data using formula
    normalized = (data - minValue) / (maxValue - minValue)
    return normalized

data = np.array([1, 3, 5, 7, 9])
normalizedData = min_max_normalize(data)

print("Data Before Normalization:", data)
print("Normalized Data:",normalizedData)

print("-"*75)


print("6. Z-Score normalization")

import numpy as np

# Z-score normalization
def z_score_normalization(data):
    mean = np.mean(data)
    std = np.std(data)
    normalized = (data - mean) / std
    return normalized


data = np.array([2, 4, 6, 8, 10])

normalized_data = z_score_normalization(data)

print("Data Before Z-Score Normalization:", data)
print("Normalized Data:", normalized_data)

print("-"*75)


print("7. Normalization by decimal scaling")

def decimal_scaling_normalization(dataList):
    max_value = max(dataList)
    num_digits = len(str(int(max_value))) - 1
    
    normalized_data = []
    for value in dataList:
        normalized_value = value / (10 ** num_digits)
        normalized_data.append(normalized_value)
    
    return normalized_data

# Example usage
dataList = [100, 250, 500, 1000]
normalized_data = decimal_scaling_normalization(dataList)
print(normalized_data)

print("-"*75)


print("8. Mean Absolute Deviation")

def mean_absolute_deviation(dataList):
    mean = sum(dataList) / len(dataList)
    deviations = [abs(value - mean) for value in dataList]
    mad = sum(deviations) / len(deviations) #mad means mean absolute value
    return mad

# Example usage
data = [6, 12, 14, 22, 11]
mad = mean_absolute_deviation(data)
print(mad)

print("-"*75)


print("9. Minkowski Distance")

def minkowskiDistance(point1, point2, p):
    distance = sum(abs(x - y) ** p for x, y in zip(point1, point2)) ** (1/p)
    return distance

# Example usage
point1 = [2, 4, 6]
point2 = [4, 7, 8]
p = 2
distance = minkowskiDistance(point1, point2, p)
print(distance)

print("-"*75)


print("10. Euclidean Distance")

import math

def euclidean_distance(point0, point1):
  sqrd_distance = sum([(a-b) ** 2 for a, b in zip(point0, point1)]) #calculating
  distance = math.sqrt(sqrd_distance) #calculating euclidean distance
  return distance

#example
point_a = [1, 2, 3]
point_b = [9, 8, 7]
distance = euclidean_distance(point_a, point_b)
print(f"Euclidean distance: {distance}" )

print("-"*75)


print("11. Manhattan Distance")

def manhattan_distance(point1, point2):
  distance = sum(abs(a - b) for a, b in zip(point1, point2)) #calculating Manhattan Distance
  return distance

#example
point_a = [1, 2, 3]
point_b = [9, 8, 7]
distance = manhattan_distance(point_a, point_b)
print(f"Manhattan distance: {distance}" )

print("-"*75)


print("12. Supremum Distance")

def supremum_distance(point1, point2):
  distance = max(abs(a - b) for a, b in zip(point1, point2)) #calculating Supremum Distance
  return distance

#example
point_a = [1, 2, 3]
point_b = [9, 8, 7]
distance = supremum_distance(point_a, point_b)
print(f"Supremum distance: {distance}" )

print("-"*75)


print("13. Cosine Similarity")

import numpy as np

def cosine_similarity(vector1, vector2):
    dot_product = np.dot(vector1, vector2)
    norm_vector1 = np.linalg.norm(vector1)
    norm_vector2 = np.linalg.norm(vector2)
    similarity = dot_product / (norm_vector1 * norm_vector2)
    return similarity

#MY Result
vector1 = np.array([1, 2, 3])
vector2 = np.array([4, 5, 6])

similarity_score = cosine_similarity(vector1, vector2)
print("Cosine similarity:", similarity_score)

print("-"*75)


print("14. Chi-Square Calculation")

import numpy as np
from scipy.stats import chi2_contingency

def calculate_chi_square(observed):
    chi2, p_value, _, _ = chi2_contingency(observed)
    return chi2, p_value

observed = np.array([[10, 15, 5], [20, 25, 15]])
chi2_statistic, p_value = calculate_chi_square(observed)
print("Chi-square statistic:", chi2_statistic)
print("p-value:", p_value)

print("-"*75)


print("15. Covariance Calculation")

import numpy as np

def calculate_covariance(x, y):
    n = len(x)
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    covariance = np.sum((x - mean_x) * (y - mean_y)) / (n - 1)
    return covariance

#use and output
x = np.array([1, 2, 3, 4, 5])
y = np.array([5, 4, 3, 2, 1])

covariance = calculate_covariance(x, y)
print("Covariance:", covariance)

print("-"*75)


print("16. Co-Variance Calculation")

import numpy as np

def calculate_covariance_matrix(data):
    n = data.shape[1]  # Number of variables
    covariance_matrix = np.cov(data, rowvar=False)
    return covariance_matrix

#Output
data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

covariance_matrix = calculate_covariance_matrix(data)
print("Covariance matrix:")
print(covariance_matrix)

print("-"*75)


print("17. Binning Methods for Data Smoothing")

import numpy as np

def equi_depth_binning(data, num_bins):
    sorted_data = np.sort(data)
    bin_edges = [sorted_data[int(i * len(sorted_data) / num_bins)] for i in range(1, num_bins)]
    binned_data = np.digitize(data, bin_edges)
    return binned_data

def bin_means(data, binned_data):
    unique_bins = np.unique(binned_data)
    bin_means = [np.mean(data[binned_data == bin]) for bin in unique_bins]
    return bin_means

def bin_boundaries(data, binned_data):
    unique_bins = np.unique(binned_data)
    bin_boundaries = [np.min(data[binned_data == bin]) for bin in unique_bins]
    bin_boundaries.append(np.max(data))
    return bin_boundaries

#All
data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
num_bins = 3


equi_depth_binned_data = equi_depth_binning(data, num_bins)
print("Equi-depth binning:")
print(equi_depth_binned_data)


means = bin_means(data, equi_depth_binned_data)
print("Bin means:")
print(means)

boundaries = bin_boundaries(data, equi_depth_binned_data)
print("Bin boundaries:")
print(boundaries)

print("-"*75)


print("18. Desicion Tree")

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# Import the dataset from CSV
dataset = pd.read_csv('Iris2.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

from sklearn.model_selection import train_test_split
# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
print(X_train) # Print the training features
print(y_train) # Print the training labels
print(X_test) # Print the test features
print(y_test) # Print the test labels

from sklearn.preprocessing import StandardScaler
# Scale the features using StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
print(X_train) # Print the scaled training features
print(X_test) # Print the scaled test features

from sklearn.tree import DecisionTreeClassifier
# Create a Decision Tree classifier
classifier = DecisionTreeClassifier(criterion = 'gini', random_state = 0)
# Fit the classifier to the training data
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
# Print the predicted and actual labels side by side
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

from sklearn.metrics import confusion_matrix, accuracy_score
# Compute and print the confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
# Compute and print the accuracy score
accuracy_score(y_test, y_pred)

from sklearn import tree

plt.figure(figsize=(15,10))
# Plot the decision tree
tree.plot_tree(classifier, filled=True)

print(tree.export_text(classifier)) # Print the text representation of the decision tree

print("-"*75)


print("19. Naïve Bayes Classifier")

from sklearn import datasets 
from sklearn import metrics 
from sklearn.naive_bayes import GaussianNB
#Loads iris dataset
dataset = datasets.load_iris ()
# Creates the Gaussian Naive Bayes classification model
model = GaussianNB ()
# Trains the model
model. fit (dataset .data, dataset. target )

# Gets real tags
expected = dataset.target 
# Makes predictions
predicted = model.predict (dataset.data)
# Prints the classification report
print (metrics.classification_report (expected, predicted)) 
# Prints the complexity matrix
print (metrics.confusion_matrix (expected, predicted) )

print("-"*75)


print("20. Classifier Evaluation Metrics: Precision and Recall, and F-measures")

from sklearn.metrics import precision_score, recall_score, f1_score

# Real tags
y_true = [0,1,0,0,1,1]

# Predicted tags
y_pred = [0,1,1,0,0,1]

#Precision

precision = precision_score(y_true, y_pred)
print("Precision:" , precision)

#Recall
recall = recall_score(y_true, y_pred)
print("Recall:" , recall)

#F-measure (F1-score)
f1 = f1_score(y_true, y_pred)
print("F-measure:",f1)

print("-"*75)

