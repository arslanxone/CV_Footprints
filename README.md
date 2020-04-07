# CV_Footprints
To detect the footprints from a skin extracted image for this project I used two models for training purpose. One to extract the foot from images and second one is to extract the edges details.

![GitHub Logo](/Images/test/1.jpeg)
Original Image

## Step 1
-> Extract foot from images through segnet model. You can use any code of segnet model from github repositories.

![GitHub Logo](/Images/Results/1.jpeg)
Mask Extracted

## Step 2
-> Used DexiNed model to get the detail of footprints

https://github.com/xavysp/DexiNed

![GitHub Logo](/Images/preda/1..png)
DexiNed model Output1

![GitHub Logo](/Images/predf/1..png)
DexiNed model Output2

## Step 3
->Merge output of both model and concatenate all outputs. You can see output in the finalprints image
![GitHub Logo](/Images/FinalPrints/1.png)
Final Output
