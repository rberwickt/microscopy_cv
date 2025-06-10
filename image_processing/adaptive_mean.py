import cv2
import matplotlib.pyplot as plt
image = cv2.imread("./ROI_snap_40x.png", cv2.IMREAD_GRAYSCALE)

image = cv2.medianBlur(image,5)

filtered_image = cv2.adaptiveThreshold(image, 255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
# Plot the original and filtered images
plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.imshow(image, cmap='gray')
plt.title('Original Image')

plt.subplot(122)
plt.imshow(filtered_image, cmap='gray')
plt.title('Adpt. Threshold Image')

plt.show()