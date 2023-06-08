# BumpiPatch

evaluate terrain <b>Bumpi</b>ness by depthmap <b>Patch</b>

## About BumpiPatch

Unlinke indoors where the driving environment is predictable, mobile robots might encounter various road conditions when it is driving outdoors. Mobile robots, which often carry sensitive equipment on their chassis or carry beverages (in the case of delivery robots) need to decide where is the stable road even within areas classsified as "traversable area". 
Therefore, by analyzing the correlation between IMU , which can digitalize the moveness of the chassis and heightmap of the road, the goal was to evaluate the driving stability by predicting "less bumpy roads" in advance. 

## Result

## Repository content :

1. source code 
  * in case of <b>Point Cloud Evaluation</b> : 
    * Heightmap Generation (pc_to_heightmap_patch.py)
    * Applicate the classification model to each patch (model_application.py)
    * Evaluate final class for each patch 
    * Colorize point cloud based on the final classes (pc_colorize.py)
  * in case of <b>Depth map Evaluation</b> : 
  * Heightmap dataset generation code
  * 
2. Media 
  * 
3. Data Files

## How to use




### Software Requirements

In order to use the provided scripts, these are the list of requirements:

```
 * Ubuntu 20.04
 * Python 3.10
 * Pytorch
 * Matplotlib
 * Open3d 0.13.0
 * OpenCV 4.5.5
```

