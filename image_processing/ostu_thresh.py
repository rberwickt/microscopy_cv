import cv2
import matplotlib.pyplot as plt
path = "image_processing/region4_FOV.PNG"

image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
image = cv2.GaussianBlur(image,(5,5),0)

#otsu
#other, filtered_image = cv2.threshold(image,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

#global threshold (for testing)
other, filtered_image = cv2.threshold(image,20,255,cv2.THRESH_BINARY)

# Plot the original and filtered images
plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.imshow(image, cmap='gray')
plt.title('Original Image')

plt.subplot(122)
plt.imshow(filtered_image, cmap='gray')
plt.title('Otsu\'s Image')

plt.show()

#only writes the images to file after you close the graph window
cv2.imwrite("image_processing/processed/otsu_thresh.jpg", filtered_image)