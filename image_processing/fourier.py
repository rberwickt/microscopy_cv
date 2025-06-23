import cv2
import matplotlib.pyplot as plt
import numpy as np
path = "image_processing/region4_FOV.PNG"

"""
After looking into filters on the fourier domain a bit more, this doesn't seem like a useful
    technique for this application, it can help remove noise but it seems like it will erase
    some particles and/or have the same problem the adaptive mean had where it segments "hollow" shapes.
    It's definitely possible to fill the donut shaped particles but 
    it's more of a pain than it's worth IMO
"""



image = cv2.imread(path, cv2.IMREAD_COLOR)
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