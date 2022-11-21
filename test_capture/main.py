import pyautogui
import base64
#import cv2 # https://www.lfd.uci.edu/~gohlke/pythonlibs/#opencv
import numpy as np
from PIL import Image

img = pyautogui.screenshot(region=[0, 0, 4, 4]) # x,y,w,h
img_data = np.asarray(img)
print("img_data=", img_data)
img_bytes = img_data.tobytes()
print("img_bytes=", img_bytes, " encode=", base64.b64encode(img_bytes))

img_encoded = base64.b64encode(img_bytes)
print("img_encoded=", img_encoded)
img_decoded = base64.b64decode(img_encoded)
print("img_decoded=", img_decoded)

# byte[]转换回ndarray
img_data2 = np.reshape(np.frombuffer(img_decoded, dtype=np.uint8), newshape=(4, 4, 3))
print("img_data2=", img_data2)

img = Image.fromarray(np.uint8(img))
img.save('screenshot3.png')
#img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR) # cvtColor用于在图像中不同的色彩空间进行转换,用于后续处理。
#cv2.imwrite('screenshot3.jpg', img)
