import cv2
import matplotlib.pyplot as plt



def thresh(path):
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    image = cv2.GaussianBlur(image,(5,5),0)

    #global threshold
    threshold = 18
    thresh_value, thresh_image = cv2.threshold(image,threshold,255,cv2.THRESH_BINARY)


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


paths = ("./dataset/region1_FOV.PNG", "./dataset/region2_FOV.PNG",
            "./dataset/region3_FOV.PNG", "./dataset/region4_FOV.png")
for path in paths:
    thresh(path)
