import cv2
import argparse

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i","--image",required=True,help="Path to the image")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])

print(image.shape)

print("width: %d pixels" % (image.shape[1]))
print("height: %d pixels" % (image.shape[0]))
print("channels: %d" % (image.shape[2]))

cv2.imshow("Image", image)
cv2.waitKey(0)

## save the image -- OpenCV handles converting filetype automatically
cv2.imwrite("newimage.jpg", image)