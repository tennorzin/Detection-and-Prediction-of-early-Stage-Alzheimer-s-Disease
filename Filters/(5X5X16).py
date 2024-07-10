import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load and preprocess the image
def preprocess_image(image_path):
    # Load image
    img = cv2.imread(image_path)
    
    # Convert image to RGB (OpenCV loads images as BGR by default)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize image to 224x224 pixels
    img_resized = cv2.resize(img_rgb, (224, 224))
    
    # Convert to grayscale
    img_gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
    
    # Normalize pixel values to range 0-1
    img_normalized = img_gray / 255.0
    
    return img_rgb, img_normalized

# Define 16 different 5x5 filters (example)
filters = [
    np.array([[1, 1, 1, 1, 1],
              [1, 1, 1, 1, 1],
              [1, 1, 1, 1, 1],
              [1, 1, 1, 1, 1],
              [1, 1, 1, 1, 1]]),
    np.array([[0, -1, 0, -1, 0],
              [-1, 5, -1, 5, -1],
              [0, -1, 0, -1, 0],
              [-1, 5, -1, 5, -1],
              [0, -1, 0, -1, 0]]),
    np.array([[-1, -1, -1, -1, -1],
              [-1, 9, 9, 9, -1],
              [-1, 9, -9, 9, -1],
              [-1, 9, 9, 9, -1],
              [-1, -1, -1, -1, -1]]),
    np.array([[1, 2, 1, 2, 1],
              [2, 4, 2, 4, 2],
              [1, 2, 1, 2, 1],
              [2, 4, 2, 4, 2],
              [1, 2, 1, 2, 1]]),
    np.array([[-1, 0, 1, 0, -1],
              [-2, 0, 2, 0, -2],
              [-1, 0, 1, 0, -1],
              [-2, 0, 2, 0, -2],
              [-1, 0, 1, 0, -1]]),
    np.array([[0, 1, 0, 1, 0],
              [1, -4, 1, -4, 1],
              [0, 1, 0, 1, 0],
              [1, -4, 1, -4, 1],
              [0, 1, 0, 1, 0]]),
    np.array([[1, 1, 0, 1, 1],
              [1, 0, -1, 0, 1],
              [0, -1, -1, -1, 0],
              [1, 0, -1, 0, 1],
              [1, 1, 0, 1, 1]]),
    np.array([[0, 0, 0, 0, 0],
              [1, 1, -1, 1, 1],
              [-1, -1, 0, -1, -1],
              [1, 1, -1, 1, 1],
              [0, 0, 0, 0, 0]]),
    np.array([[1, 0, 0, 0, 1],
              [0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0],
              [1, 0, 0, 0, 1]]),
    np.array([[-1, 0, 1, 0, -1],
              [0, 0, 0, 0, 0],
              [1, 0, -1, 0, 1],
              [0, 0, 0, 0, 0],
              [-1, 0, 1, 0, -1]]),
    np.array([[0, 1, 0, 1, 0],
              [1, 0, -1, 0, 1],
              [0, -1, 1, -1, 0],
              [1, 0, -1, 0, 1],
              [0, 1, 0, 1, 0]]),
    np.array([[0, 0, 1, 0, 0],
              [0, 0, 0, 0, 0],
              [1, 0, -1, 0, 1],
              [0, 0, 0, 0, 0],
              [0, 0, 1, 0, 0]]),
    np.array([[1, 0, 1, 0, 1],
              [0, -2, 0, -2, 0],
              [1, 0, 1, 0, 1],
              [0, -2, 0, -2, 0],
              [1, 0, 1, 0, 1]]),
    np.array([[0, 0, 0, 0, 0],
              [0, 1, 0, 1, 0],
              [0, 0, 0, 0, 0],
              [0, 1, 0, 1, 0],
              [0, 0, 0, 0, 0]]),
    np.array([[0, 0, 0, 0, 0],
              [0, 0, 1, 0, 0],
              [0, 1, -1, 1, 0],
              [0, 0, 1, 0, 0],
              [0, 0, 0, 0, 0]]),
    np.array([[1, 1, 1, 1, 1],
              [1, 1, 1, 1, 1],
              [1, 1, 1, 1, 1],
              [1, 1, 1, 1, 1],
              [1, 1, 1, 1, 1]])
]

# Example usage:
image_path = 'add_MRI_path'
original_img, preprocessed_img = preprocess_image(image_path)

# Display original and preprocessed images using matplotlib
plt.figure(figsize=(20, 15))

# Plot original image
plt.subplot(5, 5, 1)
plt.imshow(original_img)
plt.title('Original MRI Image')
plt.axis('off')

# Plot preprocessed image
plt.subplot(5, 5, 2)
plt.imshow(preprocessed_img, cmap='gray')
plt.title('Preprocessed Image')
plt.axis('off')

# Perform convolution for each filter and display convolved images
num_filters = len(filters)
for i in range(num_filters):
    # Perform convolution
    convolved_img = cv2.filter2D(preprocessed_img, -1, filters[i])
    
    # Plot each convolved image
    plt.subplot(5, 5, i + 3)
    plt.imshow(convolved_img, cmap='gray')
    plt.title(f'Filter {i + 1}')
    plt.axis('off')

plt.tight_layout()
plt.show()
