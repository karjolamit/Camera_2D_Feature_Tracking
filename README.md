# Camera_2D_Feature_Tracking

## MP.1 Data Buffer Optimization
Loading set of images using Data Buffer vector (Ring buffer). Adding new images to tail (right) and removing old images from head (left).

![Ring_Data_Buffer](https://github.com/karjolamit/Camera_2D_Feature_Tracking/blob/master/Ring_Data_Buffer.png)
```
DataFrame frame;
frame.cameraImg = imgGray;
dataBuffer.push_back(frame);
if (dataBuffer.size() > dataBufferSize) dataBuffer.pop_front();
assert(dataBuffer.size() <= dataBufferSize);
```
## MP.2 Keypoint Detection
Implement detectors like HARRIS, FAST, BRISK, ORB, AKAZE, and SIFT & enable them by setting selectable string accordingly.
```
// From the following, uncomment the one which is to be used and comment others

string detectorType = "SHITOMASI";
string detectorType = "HARRIS";
string detectorType = "FAST";
string detectorType = "BRISK";
string detectorType = "ORB";
string detectorType = "AKAZE";
string detectorType = "SIFT";
```

