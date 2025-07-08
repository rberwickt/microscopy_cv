import cv2
import matplotlib.pyplot as plt
import os


def thresh(path, value):
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    image = cv2.GaussianBlur(image,(5,5),0)

    #global threshold
    
    thresh_value, thresh_image = cv2.threshold(image,value,255,cv2.THRESH_BINARY)


    # Plot the original and filtered images
    plt.figure(figsize=(10, 5))

    plt.subplot(122)
    plt.imshow(thresh_image, cmap='gray')
    plt.title(f'Global Image ({int(thresh_value)})')

    plt.subplot(121)
    plt.imshow(image)
    plt.title('Original Image')

    plt.show()

    #only writes the images to file after you close the graph window
    cv2.imwrite(path[:-4] + ".jpg", thresh_image)

path = "./dataset"
files = []
for file in os.listdir(path):
    if file[-4:] == ".png":
        files.append(file)
        thresh(path + "/" + file, 18) # 18 seems to work pretty well
        # will likely have to tailor true masks eventually

# ['FOV1.png', 'FOV2.png', 'FOV3.png', 'FOV4.png', 'Region1_FOV (2).png', 
#  'Region2_3_FOV.png', 'Region4_FOV (2).png', 'region4_FOV.png', 'Region5_FOV.png', 
#  'Region6_FOV.png', 'Region7_FOV.png']

