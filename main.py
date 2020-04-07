import cv2
import numpy as np
import os

DATA_DIR = './Images/'

test_dir = os.path.join(DATA_DIR, 'test')
ids = sorted(os.listdir(test_dir))
orig_images_fps = [os.path.join(test_dir, image_id) for image_id in ids]

a_test_dir = os.path.join(DATA_DIR, 'pred-a')
a_ids = sorted(os.listdir(a_test_dir))
a_images_fps = [os.path.join(a_test_dir, image_id) for image_id in a_ids]

f_test_dir = os.path.join(DATA_DIR, 'pred-f')
f_ids = sorted(os.listdir(f_test_dir))
f_images_fps = [os.path.join(f_test_dir, image_id) for image_id in f_ids]

Result_dir = os.path.join(DATA_DIR, 'Results')
Results_ids = sorted(os.listdir(Result_dir))
Results_path = [os.path.join(Result_dir, image_id) for image_id in Results_ids]

for i in range(len(orig_images_fps)):
    image = cv2.imread(orig_images_fps[i])
    image_a = cv2.imread(a_images_fps[i])
    image_f = cv2.imread(f_images_fps[i])

    mask=cv2.imread(Results_path[i])

    if mask.shape[0] != image_a.shape[0]:
        mask = cv2.resize(mask, (image_a.shape[1], image_a.shape[0]))

    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    ret,mask = cv2.threshold(mask,127,255,cv2.THRESH_BINARY)

    foot = cv2.bitwise_and(image,image,mask = mask)
    foot_a = cv2.bitwise_and(image_a,image_a,mask = mask)
    foot_f = cv2.bitwise_and(image_f,image_f,mask = mask)

    footInv_a = 255 - foot_a
    footInv_f = 255 - foot_f

    Step1 = np.concatenate((image, foot), axis=1)
    Step2 = np.concatenate((Step1, footInv_a), axis=1)
    Step3 = np.concatenate((Step2, footInv_f), axis=1)

    # cv2.imshow("Original_Image", image)
    # cv2.imshow("foot", foot)
    # cv2.imshow("period_a", footInv_a)
    # cv2.imshow("period_f", footInv_f)
    # cv2.imshow("Final Output", Step3)
    # cv2.waitKey(0)

    cv2.imwrite(DATA_DIR+"/FinalPrints/"+ids[i].split(".")[0]+".png",Step3) 
    print(Step3.shape)