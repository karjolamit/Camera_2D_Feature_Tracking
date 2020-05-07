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

## MP.3 Keypoint Removal
Remove all keypoints outside of a pre-defined rectangle and only use inside points for feature tracking. Any point within 'if (vehicleRect.contains(kp.pt))' will be retained.
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
## MP.4 Keypoint Descriptors
Implement descriptors like ORB, BRIEF, HARRIS, FREAK, AKAZE, and SIFT & enable them by setting selectable string accordingly.
The implementation method is same as defining and calling a detector method in task MP.2 

```
void descKeypoints(vector<cv::KeyPoint> &keypoints, cv::Mat &img, cv::Mat &descriptors, string descriptorType)
{
   cv::Ptr<cv::DescriptorExtractor> extractor; 
   if (descriptorType.compare("BRISK") == 0)
   {
      int threshold = 30;               // FAST/AGAST detection threshold score.
      int octaves = 3;                  // detection octaves (use 0 to do single scale)
      float patternScale = 1.0f;        // apply this scale to the pattern used for sampling the neighbourhood of a keypoint.
      extractor = cv::BRISK::create(threshold, octaves, patternScale);
    }
    else if (descriptorType.compare("BRIEF") == 0)
    {
       extractor = cv::xfeatures2d::BriefDescriptorExtractor::create();
    }
    else if (descriptorType.compare("ORB") == 0)
    {
       extractor = cv::ORB::create();
    }
    else if (descriptorType.compare("FREAK") == 0)
    {
       extractor = cv::xfeatures2d::FREAK::create();
    }
    else if (descriptorType.compare("AKAZE") == 0)
    {
       extractor = cv::AKAZE::create();
    }
    else if (descriptorType.compare("SIFT") == 0)
    {
       extractor = cv::xfeatures2d::SIFT::create();
    }
    else
    {
       // when specified descriptorType is unsupported
       throw invalid_argument(descriptorType + " is not a valid descriptorType");
    }
    extractor -> compute(img, keypoints, descriptors);
}
```
## MP.5 Descriptor Matching & MP.6 Descriptor Distance Ratio
Implement FLANN matching as well as k-nearest neighbor selection. Both methods must be selectable using the respective strings in the main function. The following piece of code displays the descriptor matchers implemented in this project. 
Use the K-Nearest-Neighbor matching to implement the descriptor distance ratio test, which looks at the ratio of best vs. second-best match to decide whether to keep an associated pair of keypoints.

```
void matchDescriptors(std::vector<cv::KeyPoint> &kPtsSource, std::vector<cv::KeyPoint> &kPtsRef, cv::Mat &descSource, cv::Mat &descRef,
                      std::vector<cv::DMatch> &matches, std::string descriptorCategory, std::string matcherType, std::string selectorType)
{
    bool crossCheck = false;
    cv::Ptr<cv::DescriptorMatcher> matcher;

    // Brute force method
    if (matcherType.compare("MAT_BF") == 0)
    {
        int normType;

        // for SIFT method
        if (descriptorCategory.compare("DES_HOG") == 0)
        {
            normType = cv::NORM_L2;
        }

        // for all other binary descriptors
        else if (descriptorCategory.compare("DES_BINARY") == 0)
        {
            normType = cv::NORM_HAMMING;
        }
        
        else {
            throw invalid_argument(descriptorCategory + " is not a valid descriptorCategory");
        }

        matcher = cv::BFMatcher::create(normType, crossCheck);
    }

    // FLANN matching
    else if (matcherType.compare("MAT_FLANN") == 0)
    {
        // for SIFT method
        if (descriptorCategory.compare("DES_HOG") == 0)
        {
            matcher = cv::FlannBasedMatcher::create();
        }

        // for all other binary descriptorTypes
        else if (descriptorCategory.compare("DES_BINARY") == 0)
        {
            const cv::Ptr<cv::flann::IndexParams>& indexParams = cv::makePtr<cv::flann::LshIndexParams>(12, 20, 2);
            matcher = cv::makePtr<cv::FlannBasedMatcher>(indexParams);
        }

        else {
            throw invalid_argument(descriptorCategory + " is not a valid descriptorCategory");
        }
    }

    else {
        throw invalid_argument(matcherType + " is not a valid matcherType");
    }

    // Implementing nearest neighbor matching method (best match)
    if (selectorType.compare("SEL_NN") == 0)
    {
        matcher->match(descSource, descRef, matches);
    }
    
    // Implementing k nearest neighbors method (k=2)
    else if (selectorType.compare("SEL_KNN") == 0)
    {
        int k = 2;
        vector<vector<cv::DMatch>> knn_matches;
        matcher->knnMatch(descSource, descRef, knn_matches, k);
        
        // Filter matches using descriptor distance ratio test
        double minDescDistRatio = 0.8;
        for (auto it : knn_matches) {
            // The returned knn_matches vector contains some nested vectors with size less than
            if ( 2 == it.size() && (it[0].distance < minDescDistRatio * it[1].distance) ) {
                matches.push_back(it[0]);
            }
        }
    }
    else {
        throw invalid_argument(selectorType + " is not a valid selectorType");
    }
}
```
## Performance Evaluation 1
Count the number of keypoints on the preceding vehicle for all 10 images for all the detectors implemented.

| Detector Type | Image 0 | Image 1 | Image 2 | Image 3 | Image 4 | Image 5 | Image 6 | Image 7 | Image 8 | Image 9 | Average |
