import cv2
import matplotlib.pyplot as plt
import sys


# command line:
# python .\image_processing\processed\compare.py *IM1 FILENAME* *IM2 FILENAME*
if(len(sys.argv) < 3):
    raise RuntimeError("Less than 2 arguments to compare")
print(sys.argv)
im1_path = f"image_processing/processed/{sys.argv[1]}"
im2_path = f"image_processing/processed/{sys.argv[2]}"

# reading in grayscale (for now)
im1 = cv2.imread(im1_path, cv2.IMREAD_GRAYSCALE)
im2 = cv2.imread(im2_path, cv2.IMREAD_GRAYSCALE)

plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.imshow(im1, cmap='gray')
plt.title(sys.argv[1])

plt.subplot(122)
plt.imshow(im2, cmap='gray')
plt.title(sys.argv[2])

plt.show()