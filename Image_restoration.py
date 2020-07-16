import numpy as np
import cv2



img = cv2.imread('/home/yotam/Desktop/image_restoration/Grand_Grand_Father.jpg')

img_with_mask = cv2.imread('/home/yotam/Desktop/image_restoration/Grand_Grand_Father_mask.jpg')

# create mask

morph_size = 2
max_operator = 4
max_elem = 2
max_kernel_size = 21
morph_elem = cv2.MORPH_RECT
element = cv2.getStructuringElement(morph_elem, (2 * morph_size + 1, 2 * morph_size + 1), (morph_size, morph_size))
mask = (img_with_mask[:,:,2] > 240)*255
mask = mask.astype(np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, element)
morph_size = 4
element = cv2.getStructuringElement(morph_elem, (2 * morph_size + 1, 2 * morph_size + 1), (morph_size, morph_size))
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, element)
morph_size =2
element = cv2.getStructuringElement(morph_elem, (2 * morph_size + 1, 2 * morph_size + 1), (morph_size, morph_size))
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, element)

dst = cv2.inpaint(img,mask,3,cv2.INPAINT_NS) #cv2.INPAINT_TELEA
cv2.imshow('mask',mask)
cv2.imshow('dst',dst)
cv2.waitKey(0)
cv2.destroyAllWindows()