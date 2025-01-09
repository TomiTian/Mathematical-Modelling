import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd
from pathlib import Path
import pickle
import time

# Perform SVD on a single image and approximate it with top-r components
def apply_svd_to_image(image, r):
    """
    Takes in an image as a 1x784 array/vector and approximates it using rank-r SVD.
    Returns the reconstructed image.The higher the r the more computation effort to classify the image
    """
    # Reshape the flattened image back to 28x28, from an array of 784 elts to a 28x28 matrix
    image_2d = image.reshape(28, 28)
    
    # Perform SVD
    U, Sigma, Vt = np.linalg.svd(image_2d, full_matrices=False)
    
    # Reconstruct the image using the top-r components
    U_r = U[:, :r]
    Sigma_r = np.diag(Sigma[:r])
    Vt_r = Vt[:r, :]
    image_approx = np.dot(U_r, np.dot(Sigma_r, Vt_r))
    
    return image_approx

# Display approximations of an image with different r values
def visualize_svd(image):
    """
    Visualizes the approximation of an image with varying SVD ranks (r). Higher the r the clearer it is but higher
    """
    plt.figure(figsize=(10, 5))
    r_values = [5, 10, 20, 50, 100]  # Test different ranks
    
    for i, r in enumerate(r_values):
        # Apply SVD with rank-r components
        image_approx = apply_svd_to_image(image, r)
        
        # Display the approximation
        plt.subplot(1, len(r_values), i + 1)
        plt.imshow(image_approx, cmap='gray')
        plt.title(f'r = {r}')
        plt.axis('off')
    
    plt.show()

# Compute SVD for each digit
def compute_svd_per_digit(images, labels, r):
    svd_dict = {}
    for digit in range(10):  # digits 0-9
        class_images = images[labels == digit]  #Select images of this digit
        U, Sigma, Vt = svd(class_images.T, full_matrices=False)
        svd_dict[digit] = U[:, :r]  # Store top-r left singular vectors
    return svd_dict

# Project an image onto the subspace spanned by top-r left singular vectors
def compute_residual(image, U_r):
    """Computes the residual of the SVD matrix, distance of the image to the subspace it is spanned on (with rank r)"""
    projection = U_r @ (U_r.T @ image)
    residual = np.linalg.norm(image - projection) / np.linalg.norm(image)
    return residual

# Classify an image using the smallest residual
def classify_image(image, svd_dict):
    ''''''
    residuals = {digit: compute_residual(image, U_r) for digit, U_r in svd_dict.items()}
    return min(residuals, key=residuals.get)  # Return digit with smallest residual

def load_or_run_all_ranks(train, test, rank_range:list, folder: Path, force:bool = False):
    "Loads (from folder provided) or runs all svds for all ranks provided with the data provided"

    file_exists = (folder / "svd_file.pkl").exists()

    if not file_exists or force:
        print("SVD file does not exist or force is enabled, will run the SVD for all ranks")
        file_dict = {}

        array_t1 = []
        array_t2 =[]
        accuracy_list = []
        array_svd = []

        for i in rank_range:
            print(f"Now doing rank {i}")
            # Preparation
            s_prep = time.perf_counter()
            svd_dictionary = compute_svd_per_digit(train['x'], train['y'], r=i)
            e_prep = time.perf_counter()
            array_t1.append(e_prep-s_prep)
            
            # Prediction
            s_pred = time.perf_counter()
            predictions = np.array([classify_image(image, svd_dictionary) for image in test['x']])
            accuracy = np.mean(predictions == test['y'])
            e_pred  = time.perf_counter()
            array_t2.append(e_pred-s_pred)

            # Put all svd-dictionaries in a list
            array_svd.append(svd_dictionary)
            accuracy_list.append(accuracy)
        
        # Compile file_dict
        file_dict['array_t1'] = array_t1
        file_dict['array_t2'] = array_t2
        file_dict['accuracy_list'] = accuracy_list
        file_dict['array_svd'] = array_svd


        with open(folder / 'svd_file.pkl', 'wb') as f:
            pickle.dump(file_dict, f)
        print("SVD dictionary saved to svd_file.pkl")
    
    else:
        print("SVD file exists")
        with open(folder / 'svd_file.pkl', 'rb') as f:
            file_dict = pickle.load(f)
        print("SVD file loaded")
    
    # Unpack the file_dict, to return the individual parts.
    return file_dict['array_t1'], file_dict['array_t2'], file_dict['accuracy_list'], file_dict['array_svd']



