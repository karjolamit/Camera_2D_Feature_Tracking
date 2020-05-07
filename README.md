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
Following is an example of defining Shitomasi detector function. The other types of detectors are also defined on similar lines.

```
void detKeypointsShiTomasi(vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    // compute detector parameters based on image size
    int blockSize = 4;                      // avg blocksize for computing a derivative covariation matrix over each pixel neighborhood
    double maxOverlap = 0.0;                // max permissible overlap between two features in %
    double minDistance = (1.0 - maxOverlap) * blockSize;
    int maxCorners = img.rows * img.cols / max(1.0, minDistance);     // max number of keypoints
    double qualityLevel = 0.01;                                       // minimal accepted quality of image corners
    double k = 0.04;

    // Apply corner detection
    double t = (double)cv::getTickCount();
    vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(img, corners, maxCorners, qualityLevel, minDistance, cv::Mat(), blockSize, false, k);

    // add corners to result vector
    for (auto it = corners.begin(); it != corners.end(); ++it)
    {
        cv::KeyPoint newKeyPoint;
        newKeyPoint.pt = cv::Point2f((*it).x, (*it).y);
        newKeyPoint.size = blockSize;
        keypoints.push_back(newKeyPoint);
    }
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    // cout << "Shi-Tomasi detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    // visualize results
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "Shi-Tomasi Corner Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
}
```
Using following, any of the detectors can be used by selcting 'detetorType' 

```
// Shi-Tomasi
if (detectorType.compare("SHITOMASI") == 0)
{
  detKeypointsShiTomasi(keypoints, imgGray, false);
}
// Harris
else if (detectorType.compare("HARRIS") == 0)
{
  detKeypointsHarris(keypoints, imgGray, false);
}
// Modern detector types, including FAST, BRISK, ORB, AKAZE, and SIFT
else if (detectorType.compare("FAST")  == 0 ||
         detectorType.compare("BRISK") == 0 ||
         detectorType.compare("ORB")   == 0 ||
         detectorType.compare("AKAZE") == 0 ||
         detectorType.compare("SIFT")  == 0)
 {
   detKeypointsModern(keypoints, imgGray, detectorType, false);
 }
 // when specified detectorType is unsupported
 else
 {
   throw invalid_argument(detectorType + " is not a valid detectorType");
 }
```

## MP.2 Keypoint Removal
Remove all keypoints outside of a pre-defined rectangle and only use inside points for feature tracking. 
```
// Hardcoded bounding box to keep only keypoints on the preceding vehicle (Region of Interest)
bool bFocusOnVehicle = true;
cv::Rect vehicleRect(535, 180, 180, 150);
if (bFocusOnVehicle)
{
   vector<cv::KeyPoint> filteredKeypoints;
   for (auto kp : keypoints) 
   {
       if (vehicleRect.contains(kp.pt)) filteredKeypoints.push_back(kp);
   }
   keypoints = filteredKeypoints;
}
```
