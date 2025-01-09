import numpy as np
import struct
import matplotlib.pyplot as plt
import random
import time
from statistics import multimode
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Load images from the MNIST dataset
def load_image_file(filename):
    with open(filename, 'rb') as f:
        # Skip the magic number and number of images
        _, num_images = struct.unpack(">II", f.read(8))
        rows, cols = struct.unpack(">II", f.read(8))  # Image dimensions
        # Read image data and reshape into (num_images, rows*cols)
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_images, rows * cols)
    return {'n': num_images, 'x': images}

# Load labels from the MNIST dataset
def load_label_file(filename):
    with open(filename, 'rb') as f:
        # Skip the magic number and number of labels
        _, num_labels = struct.unpack(">II", f.read(8))
        # Read label data
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels

# Load training and test datasets
def load_mnist():
    train = load_image_file('C:/0_Mano/Programming/VSCode Projects/Mathematical Modelling/train-images.idx3-ubyte')
    test = load_image_file('C:/0_Mano/Programming/VSCode Projects/Mathematical Modelling/t10k-images.idx3-ubyte')
    
    train['y'] = load_label_file('C:/0_Mano/Programming/VSCode Projects/Mathematical Modelling/train-labels.idx1-ubyte')
    test['y'] = load_label_file('C:/0_Mano/Programming/VSCode Projects/Mathematical Modelling/t10k-labels.idx1-ubyte')
    
    return train, test

# Display an MNIST digit image
def show_digit(arr784):
    plt.imshow(arr784.reshape(28, 28), cmap='gray')
    plt.axis('off')  # Hide axes for cleaner display
    plt.show()

# Euclidean distance between two images
def euclid_dist(img_1, img_2):
    distance = 0
    for i in range(len(img_1)):
        distance += (int(img_1[i]) - int(img_2[i]))**2
    euclid_dist = np.sqrt(distance)
    return euclid_dist

# Not used anywhere in the end; only here for fun
def manhattan_dist(img_1, img_2):
    distance = 0
    for i in range(len(img_1)):
        distance += abs(int(img_1[i]) - int(img_2[i]))
    return distance

# Compute the average image of each digit (returns 10 images, one for each digit) and their labels
def average_images(training_img_set, training_label_set):
    avg_images = []
    avg_images_labels = [i for i in range(10)]
    for i in range(10):
        sum_images = [0] * 784
        digit_count = 0
        for j in range(len(training_img_set)):
            if (training_label_set[j] == i):
                digit_count += 1
                for k in range(784):
                    sum_images[k] += int(training_img_set[j][k])
        avg_digit = [pixel/digit_count for pixel in sum_images]
        avg_images.append(avg_digit)
    return (avg_images, avg_images_labels)

# For each digit, takes all images of that digit and groups them into n clusters; returns the cluster centroids (meaning 10*n total images) and their labels
def n_centroid_images_each(training_img_set, training_label_set, n):
    centroids= []
    centroids_labels = []
    for i in range(10):
        digit_images = [training_img_set[j] for j in range(len(training_img_set)) if training_label_set[j] == i]
        kmeans = KMeans(n_clusters = n).fit(digit_images)
        digit_centroids = kmeans.cluster_centers_
        centroids.extend(digit_centroids)
        centroids_labels.extend([i] * n)
    return(centroids, centroids_labels)

# The KNN algorithm: given known sets (images and labels), a single image, and the number k of neighbors to consider, returns the predicted digit in the image.
# The "known sets" may be the entire training dataset, the average images, the cluster centroid images, etc.
def k_nearest_neighbors(known_img_set, known_label_set, img, k):
    distances_list = [euclid_dist(i, img) for i in known_img_set]
    distances_array = np.array(distances_list)
    neighbor_indices = np.argpartition(distances_array, k-1)[0:k].tolist()
    neighbors = [known_label_set[i] for i in neighbor_indices]
    #print(neighbors)
    digits = multimode(neighbors)
    if (len(digits) == 1):
        digit = digits[0]
    else:
        closest_index = distances_list.index(min(distances_list))
        digit = known_label_set[closest_index]
    return digit

# Load MNIST data
train, test = load_mnist()

setup_start = time.perf_counter()
# Select one line based on the method used:
#known_images, known_images_labels = train['x'], train['y']
#known_images, known_images_labels = train['x'][0:2000], train['y'][0:2000]
#known_images, known_images_labels = average_images(train['x'], train['y'])
known_images, known_images_labels = n_centroid_images_each(train['x'], train['y'], 5)
setup_end = time.perf_counter()
setup_time = setup_end - setup_start
print("Setup time: " + str(setup_time))

predictions = []
true_values = [test['y'][i] for i in range(4000, 6000)]
correctness_list = []

exec_start = time.perf_counter()
for i in range(4000, 6000):
    guess = k_nearest_neighbors(known_images, known_images_labels, test['x'][i], 1)
    predictions.append(guess)
    if (guess == test['y'][i]):
        correctness_list.append(1)
    else: correctness_list.append(0)
exec_end = time.perf_counter()
exec_time = exec_end - exec_start
print("Execution time: " + str(exec_time))

accuracy = sum(correctness_list) / len(correctness_list)
print("Accuracy is " + str(accuracy))

conf_matrix = confusion_matrix(true_values, predictions)
disp = ConfusionMatrixDisplay(confusion_matrix = conf_matrix, display_labels = [str(i) for i in range(10)])
disp.plot(cmap = plt.cm.Blues)
plt.title("Clusters, n = 5, Range (4000, 6000)")
plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.show()

### END OF CODE ###

# These functions are obsolete: they are ideas tried during the research process which were ultimately scrapped. Leaving them here for anyone curious.

def n_random_images_each(training_img_set, training_label_set, n):
    random_images = []
    random_images_labels = []
    for i in range(10):
        digit_images = [training_img_set[j] for j in range(len(training_img_set)) if training_label_set[j] == i]
        random_digit_images = random.choices(digit_images, k = n)
        random_images.extend(random_digit_images)
        random_images_labels.extend([i] * n)
    return random_images, random_images_labels

def n_quantiled_images_each(training_img_set, training_label_set, n):
    quantiled_images = []
    quantiled_images_labels = []
    averages = average_images(training_img_set, training_label_set)
    for i in range(10):
        digit_images = [training_img_set[j] for j in range(len(training_img_set)) if training_label_set[j] == i]
        num_images = len(digit_images)
        distances_from_avg = [euclid_dist(img, averages[i]) for img in digit_images]
        sorted_indices = np.argsort(distances_from_avg)
        sorted_digit_images = [digit_images[j] for j in sorted_indices]
        quantiled_digit_images = [sorted_digit_images[num_images * j // n] for j in range(n)]
        quantiled_images.extend(quantiled_digit_images)
        quantiled_images_labels.extend([i] * n)
    return quantiled_images, quantiled_images_labels

def n_quantiled_averages_each(training_img_set, training_label_set, n):
    quantiled_averages = []
    quantiled_averages_labels = []
    averages = average_images(training_img_set, training_label_set)
    for i in range(10):
        digit_images = [training_img_set[j] for j in range(len(training_img_set)) if training_label_set[j] == i]
        num_images = len(digit_images)
        distances_from_avg = [euclid_dist(img, averages[i]) for img in digit_images]
        sorted_indices = np.argsort(distances_from_avg)
        sorted_digit_images = [digit_images[j] for j in sorted_indices]
        quantiled_digit_averages = []
        for j in range(n):
            quantiled_subset = sorted_digit_images[(num_images * j // n) : (num_images * (j+1) // n)]
            sum_images = [0] * 784
            for k in range(len(quantiled_subset)):
                for l in range(784):
                    sum_images[l] += quantiled_subset[k][l]
            avg_digit = [pixel/len(quantiled_subset) for pixel in sum_images]
            quantiled_digit_averages.append(avg_digit)
        quantiled_averages.extend(quantiled_digit_averages)
        quantiled_averages_labels.extend([i] * n)
    return quantiled_averages, quantiled_averages_labels

# Here are code snippets in case you want to visualize the average images for each digit, or the cluster centroids for some particular digit

"""averages = average_images(train['x'], train['y'])
for i in range(10):
    img_arr = np.array(averages[i])
    img = np.reshape(img_arr, 28*28)
    show_digit(img)"""

"""data = [train['x'][i] for i in range(len(train['x'])) if train['y'][i] == 3]
kmeans = KMeans(n_clusters = 5).fit(data)
centers = kmeans.cluster_centers_
for center in centers:
    show_digit(center)"""
