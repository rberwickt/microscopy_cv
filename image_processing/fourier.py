import cv2
import matplotlib.pyplot as plt
import numpy as np
image = cv2.imread("./ROI_snap_40x.png", cv2.IMREAD_COLOR)
# UNFINISHED
filter = np.array([[1,1,1],
                  [1,1,1],
                  [1,1,1]])
filtered_image = cv2.filter2D(src=image, ddepth=-1, kernel=filter)
# Plot the original and filtered images
plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.imshow(image, cmap='gray')
plt.title('Original Image')

plt.subplot(122)
plt.imshow(filtered_image, cmap='gray')
plt.title('1s Convolution')

plt.show()