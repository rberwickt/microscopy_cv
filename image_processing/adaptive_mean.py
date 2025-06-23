import cv2
import matplotlib.pyplot as plt
path = "image_processing/region4_FOV.png"

image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

image = cv2.medianBlur(image,5) #removes a lot of noise (and possibly particles)

# a higher C value (last argument) is a more aggressive threshold
# block size (second to last) must be > 1 and odd
#adaptive mean
mean_filtered_image = cv2.adaptiveThreshold(image, 255,cv2.ADAPTIVE_THRESH_MEAN_C,
                                            cv2.THRESH_BINARY_INV,15,3)
#adaptive gaussian
gaussian_filtered_image = cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                cv2.THRESH_BINARY_INV,15,3)

# blur testing

#gaussian_filtered_image = cv2.GaussianBlur(gaussian_filtered_image, (3, 3), 0)
#mean_filtered_image = cv2.GaussianBlur(mean_filtered_image, (3, 3), 0)

# Plot the original and filtered images
plt.figure(figsize=(10, 5))
plt.subplot(131)
plt.imshow(image, cmap='gray')
plt.title('Original Image')

plt.subplot(132)
plt.imshow(mean_filtered_image, cmap='gray')
plt.title('Mean Adpt. Threshold')

plt.subplot(133)
plt.imshow(gaussian_filtered_image, cmap='gray')
plt.title('Gaussian Adpt. Threshold')

plt.show()

#only writes the images to file after you close the graph window
cv2.imwrite("image_processing/processed/adaptive_mean.jpg", mean_filtered_image)
cv2.imwrite("image_processing/processed/adaptive_gaussian.jpg", gaussian_filtered_image)