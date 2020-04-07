import cv2
import numpy as np
import os

DATA_DIR = './Input_Images/'

x_test_dir = os.path.join(DATA_DIR, 'output')
ids = sorted(os.listdir(x_test_dir))
images_fps = [os.path.join(x_test_dir, image_id) for image_id in ids]

Result_dir = os.path.join(DATA_DIR, 'Results')
Results_ids = sorted(os.listdir(Result_dir))
Results_path = [os.path.join(Result_dir, image_id) for image_id in Results_ids]

print(len(images_fps))
print(len(Results_path))
#exit()

for i in range(len(images_fps)):
	#print()
    image = cv2.imread(images_fps[i])
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # image = cv2.equalizeHist(image) 

    mask=cv2.imread(Results_path[i])

    if mask.shape[0] != image.shape[0]:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]))

    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    ret,mask = cv2.threshold(mask,127,255,cv2.THRESH_BINARY)

    foot = cv2.bitwise_and(image,image,mask = mask)

    # if foot.shape[0] > 1000 or foot.shape[1] > 1000:
    #     fact = 0.6
    #     foot = cv2.resize(foot, (foot.shape[1], foot.shape[0]), fact, fact, cv2.INTER_CUBIC)

    foot = cv2.cvtColor(foot, cv2.COLOR_BGR2GRAY)
    ret2, footInv = cv2.threshold(foot,127,255,cv2.THRESH_BINARY_INV)

    cv2.imwrite(DATA_DIR+"/FinalPrints/"+ids[i].split(".")[0]+".png",footInv) 

    print(foot.shape)