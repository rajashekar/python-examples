import cv2
image = cv2.imread("heapdump-1.png")
print(image.shape)
cv2.imshow("Image", image)
cv2.waitKey(0)
