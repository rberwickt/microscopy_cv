import cv2
import matplotlib.pyplot as plt
image = cv2.imread("./ROI_snap_40x.png", cv2.IMREAD_GRAYSCALE)

image = cv2.GaussianBlur(image,(5,5),0)

other, filtered_image = cv2.threshold(image,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# Plot the original and filtered images
plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.imshow(image, cmap='gray')
plt.title('Original Image')

plt.subplot(122)
plt.imshow(filtered_image, cmap='gray')
plt.title('Otsu\'s Image')

plt.show()