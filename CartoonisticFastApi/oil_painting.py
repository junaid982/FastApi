import cv2
img = cv2.imread('./bhandup.jpg')
res = cv2.xphoto.oilPainting(img, 2, 1)

cv2.imshow("original", img)
cv2.imshow("res", res)
cv2.waitKey(0)
cv2.destroyAllWindows()