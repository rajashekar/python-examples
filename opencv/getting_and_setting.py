import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i","--image",required=True,help="Path to the image")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
(h, w) = image.shape[:2]
cv2.imshow("Original", image)

# images are just Numpy arrays. The top-left pixel can be found at (0,0)
(b, g, r) = image[0, 0]
print("Pixel at (0,0) - Red:{r}, Green: {g}, Blue: {b}".format(r=r, g=g, b=b))

# x=111, y=225
(b, g, r) = image[225, 111]
print("Pixel at (111,225) - Red:{r}, Green: {g}, Blue: {b}".format(r=r, g=g, b=b))

# now, lets change the value of the pixel at (0,0) and make it red
image[0, 0] = (0, 0, 255)
(b, g, r) = image[0,0]
print("Pixel at (0,0) - Red:{r}, Green: {g}, Blue: {b}".format(r=r, g=g, b=b))

# compute the center of the image, which is simply the width and height
# divided by 2
(cX, cY) = (w // 2, h // 2)

# since we are using NumPy arrays, we can apply slicing and grab large chunks
# of the image -- lets grab the top-left corner

tl = image[0:cY, 0:cX]
cv2.imshow("Top-Left Corner", tl)

# in a similar fashion, let's grab the top-right, bottom-right, and bottom-left
# corners and display them

tr = image[0:cY, cX:w]
br = image[cY:h, cX:w]
bl = image[cY:h, 0:cX]

cv2.imshow("Top-Right Corner", tr)
cv2.imshow("Bottom-Right Corner", br)
cv2.imshow("Bottom-Left Corner", bl)

# now lets make the top-left corner of the original image green
image[0:cY, 0:cX] = (255, 0, 0)

cv2.imshow("Updated", image)

cv2.waitKey(0)