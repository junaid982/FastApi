import sys
import os
import cv2
import easygui
import numpy as np
import matplotlib.pyplot as plt
import imageio


ImagePath = easygui.fileopenbox()

img  =cv2.imread(ImagePath)

if img is None:
    print('Can not find image Choose Image')
    sys.exit()

img = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)
img_grey = cv2.cvtColor(img , cv2.COLOR_RGB2GRAY)
img_blur = cv2.medianBlur(img_grey,3)

plt.imshow(img_blur)
plt.axis('off')
plt.title("Green")
plt.show()


img_style = cv2.stylization(img , sigma_s=150 , sigma_r=0.25)

plt.imshow(img_style)
plt.axis('off')
plt.title("Cartoon")
plt.show()

img_style = cv2.stylization(img , sigma_s=700 , sigma_r=0.55)

plt.imshow(img_style)
plt.axis('off')
plt.title("Oil Paintings")
plt.show()


edge = cv2.adaptiveThreshold(img_blur,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,3,3)
plt.imshow(edge)
plt.axis('off')
plt.title("Edge Mask")
plt.show()

img_bb = cv2.bilateralFilter(img_blur,15,75,75)
plt.imshow(img_bb)
plt.axis('off')
plt.title("Filter Green Color")
plt.show()



