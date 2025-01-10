import numpy as np
import random
from statistics import multimode
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

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




def show_digits(images, rows=1, cols=None):
    """
    Display multiple MNIST digit images in a grid.

    Parameters:
    - images: List or array of 28x28 images.
    - rows: Number of rows in the grid.
    - cols: Number of columns in the grid (optional, calculated if None).
    """
    if cols is None:
        cols = len(images) // rows if len(images) % rows == 0 else (len(images) // rows + 1)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    axes = axes.flatten()  # Flatten in case of a single row or column

    for i, ax in enumerate(axes):
        if i < len(images):
            ax.imshow(images[i], cmap='gray')
            ax.axis('off')  # Hide axes for a cleaner display
        else:
            ax.axis('off')  # Hide unused subplots

    plt.tight_layout()
    plt.show()


