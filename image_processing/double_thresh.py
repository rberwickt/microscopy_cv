import cv2
import matplotlib.pyplot as plt
path = "image_processing/region4_FOV.png"

image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
image = cv2.GaussianBlur(image,(5,5),0)

#otsu
otsu_value, otsu_image = cv2.threshold(image,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

#global threshold
threshold = 18
thresh_value, thresh_image = cv2.threshold(image,threshold,255,cv2.THRESH_BINARY)

# masking global for later processing
color = cv2.imread(path, cv2.IMREAD_COLOR)
masked = cv2.bitwise_and(color, color, mask=thresh_image)

# Plot the original and filtered images
plt.figure(figsize=(10, 5))


plt.subplot(133)
plt.imshow(otsu_image, cmap='gray')
plt.title(f'Otsu\'s Image ({int(otsu_value)})')

plt.subplot(132)
plt.imshow(thresh_image, cmap='gray')
plt.title(f'Global Image ({int(thresh_value)})')

plt.subplot(131)
plt.imshow(image, cmap='gray')
plt.title('Original Image')

plt.show()

#only writes the images to file after you close the graph window
cv2.imwrite("image_processing/processed/otsu_thresh.jpg", otsu_image)
cv2.imwrite("image_processing/processed/global_thresh.jpg", thresh_image)
cv2.imwrite("image_processing/processed/global_thresh_masked.jpg", masked)