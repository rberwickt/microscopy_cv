import cv2
import matplotlib.pyplot as plt

path = "image_processing/region4_FOV.png"

image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
image = cv2.GaussianBlur(image, (3, 3), 0)
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
filtered_image = cv2.Laplacian(image_gray, cv2.CV_16S, ksize=3)
# Plot the original and filtered images
plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.imshow(image, cmap='gray')
plt.title('Original Image')

plt.subplot(122)
plt.imshow(filtered_image, cmap='gray')
plt.title('LoG Filtered Image')

plt.show()

#only writes the images to file after you close the graph window
cv2.imwrite("image_processing/processed/laplace_gaussian.jpg", filtered_image)