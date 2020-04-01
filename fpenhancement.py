from .ridgefilter import RidgeFilter
from .normalizer import Normalizer
from .ridgeorient import RidgeOrient

import cv2
import numpy as np

class FpEnhancement:

    normalizedImage = None
    orientationImage = None
    
    def __init__(self, windowSize, thresh, orientsmoothsigma, 
                blocksigma, gradientsigma, kx, ky):

        self.windowSize = 38;
        self.thresh = 0.000000001;
        self.orientsmoothsigma = 5.0;
        self.blocksigma = 5.0;
        self.gradientsigma = 1.0;
        self.kx = 0.8;
        self.ky = 0.8;

    def run(self, inputImage):
        blurredImage =  cv2.medianBlur(inputImage, (3,3))

        if len(blurredImage.shape) != 1:
            blurredImage = cv2.cvtColor(blurredImage, cv2.COLOR_BGR2GRAY)
    
        # Perform normalization using the method provided in the paper
        self.normalizedImage = Normalizer.run(blurredImage, 0, 1)

        # Calculate ridge orientation field
        self.orientationImage = RidgeOrient.run(normalizedImage, gradientsigma, blocksigma, orientsmoothsigma)

        # The frequency is set to 0.11 experimentally. You can change this value
        # or use a dynamic calculation instead.
        freq = np.ones(normalizedImage.rows, normalizedImage.cols, normalizedImage.type())
        freq *= 0.11

        # Get the final enhanced image and return it as result
        enhancedImage = RidgeFilter.run(normalizedImage, orientationImage, freq, kx, ky)

        return enhancedImage

    def postProcessingFilter(self, inputImage):
        lowThreshold = 10
        ratio = 3
        kernel_size = 3

        if len(inputImage.shape) != 1:
            inputImageGrey = cv2.cvtColor(inputImage, cv2.COLOR_BGR2GRAY)
        else:
            inputImageGrey = inputImage
        
        # Blurring the image several times with a kernel 3x3 to have smooth surfaces
        for j in range(30):
            inputImageGrey = cv2.blur(inputImageGrey, (3, 3))
        
        # Canny detector to catch the edges
        filter = cv2.Canny(inputImageGrey, lowThreshold, lowThreshold * ratio, kernel_size)
        
        #  Use Canny's output as a mask
        processedImage = cv2.Scalar.all(0);
        inputImageGrey.copyTo(processedImage, filter);

        dilationSize = 10;
        dilationType = 1;
        element = cv2.getStructuringElement(dilationType,cv2.Size(2 * dilationSize + 1, 2 * dilationSize + 1), cv2.Point(dilationSize, dilationSize))

        # Dilate the image to get the contour of the finger
        cv2.dilate(processedImage, processedImage, element);

        # Fill the image from the middle to the edge.
        cv2.floodFill(processedImage, cv2.Point(filter.cols / 2, filter.rows / 2), cv2.Scalar(255))

        return processedImage
    
    def getNormalizedImage(self):
        return self.normalizedImage
    
    def getOrientationImage(self):
        return self.orientationImage