import cv2
import matplotlib.pyplot as plt
import numpy as np
# !!!!!!!!!!!!!!!!!!!!!!
# only removes the largest connected components, does no further masking (as opposed to other file)
# !!!!!!!!!!!!!!!!!!!!!!
path = "image_processing/processed/adaptive_mean2.jpg"
masked_path = "image_processing/region1_FOV.png"

original_image = cv2.imread(masked_path, cv2.IMREAD_COLOR)
image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

(totalLabels, label_ids, values, centroid) = cv2.connectedComponentsWithStats(image,
                                            4, cv2.CV_32S)

print(totalLabels)

# range 0,5 since there are 5 statistics for each component (detailed on opencv documentation)
areas = np.take(values,range(4,5),axis=1)
areas = areas.flatten()
print(areas)
threshold = np.mean(areas)
print(f"AREA THRESHOLD: {threshold}")

# make the entire image white (to specifically remove the large components)
shape = image.shape
cv2.rectangle(image, (0,0), (shape[1] - 1, shape[0] - 1), 255, cv2.FILLED)

plt.figure(figsize=(10, 5))

plt.imshow(image)
plt.title(f'Fullwhite {shape} -> {(0,0)}/{(shape[1] - 1, shape[0] - 1)}')
plt.show()

for i in range(1,totalLabels):
    area = areas[i]
    if area > threshold: # detecting very large shapes
        #print(f"Large shape: {area}")
        # getting bounding box
        x = values[i, cv2.CC_STAT_LEFT]
        y = values[i, cv2.CC_STAT_TOP]
        width = values[i, cv2.CC_STAT_WIDTH]
        height = values[i, cv2.CC_STAT_HEIGHT]

        # removing
        cv2.rectangle(image, (x,y), (x+width,y+height), 0, cv2.FILLED)



# masking 
masked = cv2.bitwise_and(original_image, original_image, mask=image)


# display image

plt.figure(figsize=(10, 5))

plt.subplot(133)
plt.imshow(masked)
plt.title('Masked Image')

plt.subplot(132)
plt.imshow(image, cmap="gray")
plt.title('Filled Mask')

plt.subplot(131)
plt.imshow(original_image)
plt.title('Original Image')

plt.show()

# writing image (only writes after display is closed)
cv2.imwrite("image_processing/processed/removed2.jpg", masked)
cv2.imwrite("image_processing/processed/removed_mask2.jpg", image)
