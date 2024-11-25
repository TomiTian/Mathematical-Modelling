import numpy as np
import struct
import matplotlib.pyplot as plt
import vptree
from statistics import mode

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
    train = load_image_file('D:/Programming/VSCode Projects/Mathematical Modelling/train-images.idx3-ubyte')
    test = load_image_file('D:/Programming/VSCode Projects/Mathematical Modelling/t10k-images.idx3-ubyte')
    
    train['y'] = load_label_file('D:/Programming/VSCode Projects/Mathematical Modelling/train-labels.idx1-ubyte')
    test['y'] = load_label_file('D:/Programming/VSCode Projects/Mathematical Modelling/t10k-labels.idx1-ubyte')
    
    return train, test

# Display an MNIST digit image
def show_digit(arr784):
    plt.imshow(arr784.reshape(28, 28), cmap='gray')
    plt.axis('off')  # Hide axes for cleaner display
    plt.show()


def euclid_dist(img_1, img_2):
    distance = 0
    for i in range(len(img_1) - 1):
        distance += (int(img_1[i]) - int(img_2[i]))**2
    euclid_dist = np.sqrt(distance)
    return euclid_dist

def manhattan_dist(img_1, img_2):
    distance = 0
    for i in range(len(img_1)):
        distance += (int(img_1[i]) - int(img_2[i]))
    return distance

def augment_images(training_img_set, training_label_set):
    augm_list = []
    for i in range(len(training_img_set)):
        image = training_img_set[i].tolist()
        image.append(int(training_label_set[i]))
        augm_list.append(image)
    return augm_list

def average_images(training_img_set, training_label_set):
    avg_images = []
    for i in range(10):
        sum_images = [0] * 784
        digit_count = 0
        for j in range(len(training_img_set)):
            if (training_label_set[j] == i):
                for k in range(784):
                    sum_images[k] += training_img_set[j][k]
                    digit_count += 1
        avg_digit = [l/digit_count for l in sum_images]
        avg_images.append(avg_digit)
    return avg_images

def k_nearest_neighbors_old(training_img_set, training_label_set, img, k):
    distances_list = [euclid_dist(i, img) for i in training_img_set]
    distances_array = np.array(distances_list)
    neighbor_indices = np.argpartition(distances_array, k-1)[0:k].tolist()
    neighbors = [training_label_set[i] for i in neighbor_indices]
    print(neighbors)
    digit = mode(neighbors)
    return digit

def k_nearest_neighbors(tree: vptree, img, k):
    neighbor_images = tree.get_n_nearest_neighbors(img, k)
    neighbor_images_cleaned = [neighbor[1].tolist() for neighbor in neighbor_images]
    neighbors = [image[-1] for image in neighbor_images_cleaned]
    print(neighbors)
    digit = mode(neighbors)
    return digit

# Load MNIST data
train, test = load_mnist()

# Display an example image (e.g., the 5th training image)
#show_digit(test['x'][1])

#guess = k_nearest_neighbors(train['x'], train['y'], test['x'][1], 3)
#print(guess)
"""correctness_list = []
for i in range(300, 370):
    guess = k_nearest_neighbors_old(train['x'][0:5000], train['y'][0:5000], test['x'][i], 3)
    print(guess)
    if (guess == test['y'][i]):
        correctness_list.append(1)
    else: correctness_list.append(0)

accuracy = sum(correctness_list) / len(correctness_list)
print("Accuracy is " + str(accuracy))"""
"""train_augmented = augment_images(train['x'][0:5000], train['y'][0:5000])
print("augmented yay")
tree = vptree.VPTree(train_augmented, euclid_dist)
print("tree treed tety")
for i in range(0, 20):
    imagee = test['x'][i].tolist()
    guess = k_nearest_neighbors(tree, imagee, 3)
    print(guess)"""
averages = average_images(train['x'], train['y'])
guess = k_nearest_neighbors_old(averages, [i for i in range(10)], test['x'][0], 3)
print(guess)
print(test['y'][0])
