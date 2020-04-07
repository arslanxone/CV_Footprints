# from .fpenhancement import FpEnhancement
import cv2
import numpy as np
import os

DATA_DIR = './Input_Images/'

x_test_dir = os.path.join(DATA_DIR, 'test')
ids = sorted(os.listdir(x_test_dir))
images_fps = [os.path.join(x_test_dir, image_id) for image_id in ids]

Result_dir = os.path.join(DATA_DIR, 'Results')
Results_ids = sorted(os.listdir(Result_dir))
Results_path = [os.path.join(Result_dir, image_id) for image_id in Results_ids]

for i in range(len(x_test_dir)):
    image = cv2.imread(images_fps[i])
    
    mask=cv2.imread(Results_path[i])
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    ret,mask = cv2.threshold(mask,127,255,cv2.THRESH_BINARY)

    foot = cv2.bitwise_and(image,image,mask = mask)

    if foot.shape[0] > 1000 or foot.shape[1] > 1000:
        fact = 0.6
        foot = cv2.resize(foot, (foot.shape[1], foot.shape[0]), fact, fact, cv2.INTER_CUBIC)

    cv2.imwrite(DATA_DIR+"/FinalPrints/"+ids[i].split(".")[0]+".png",foot) 
    # # Run the enhancement algorithm
	# enhancedImage = FpEnhancement.run(foot);

	# # Doing the postProcessing
	# filter = FpEnhancement.postProcessingFilter(foot);

	# # Finally applying the filter to get the end result
	# endRes = cv2.Scalar.all(0);
	# enhancedImage.copyTo(endRes, filter);
	# cv2.imshow("endRes", endRes  );
	# cv2.waitKey(0);

    # cv2.imshow("foot", foot)
    # cv2.waitKey(0)