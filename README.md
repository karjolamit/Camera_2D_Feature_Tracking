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
## MP.7 Performance Evaluation 1
Count the number of keypoints on the preceding vehicle for all 10 images for all the detectors implemented.

| Detector Type | Image 0 | Image 1 | Image 2 | Image 3 | Image 4 | Image 5 | Image 6 | Image 7 | Image 8 | Image 9 | Average | Neighborhood Size |
| ------------- | ------- | ------- | ------- | ------- | ------- | ------- | ------- | ------- | ------- | ------- | ------- | ------------------------ |
| SHITOMASI | 125 | 118 | 123 | 120 | 120 | 113 | 114 | 123 | 111 | 112 | 108 | 4 |
| HARRIS | 17 | 14 | 19 | 22 | 26 | 47 | 18 | 33 | 27 | 35 | 26 | 6 |
| FAST | 419 | 427 | 404 | 423 | 386 | 414 | 418 | 406 | 396 | 401 | 434 | 7 |
| BRISK | 264 | 282 | 282 | 277 | 297 | 279 | 289 | 272 | 266 | 254 | 276 | 8.4 - 16.5484 |
| ORB | 92 | 102 | 106 | 113 | 109 | 125 | 130 | 129 | 127 | 128 | 116 | 31 |
| AKAZE | 166 | 157 | 161 | 155 | 163 | 164 | 173 | 175 | 177 | 179 | 168 | 8 |
| SIFT | 138 | 132 | 124 | 137 | 134 | 140 | 137 | 148 | 159 | 137 | 139 | 1.89 - 5.10 |

From above table, it is clearly evident that the FAST developed highest number of keypoint detections whereas HARRIS developed lowest number of keypoint detections.

## MP.8 Performance Evaluation 2
Count the number of matched keypoints for all 10 images using all possible combinations of detectors and descriptors. In the matching step, the BF approach is used with the descriptor distance ratio set to 0.8.

| Detector Type | Descriptor Type | NMP Img 0 | NMP Img 1 | NMP Img 2 | NMP Img 3 | NMP Img 4 | NMP Img 5 | NMP Img 6 | NMP Img 7 | NMP Img 8 |
| ------------- | --------------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- |
| SHITOMASI | BRISK | 95 | 88 | 80 | 90 | 82 | 79 | 85 | 86 | 82 | 
| HARRIS | BRISK | 12 | 10 | 14 | 16 | 16 | 17 | 15 | 22 | 21 |
| FAST | BRISK | 256 | 243 | 241 | 239 | 215 | 251 | 248 | 243 | 247 |
| BRISK | BRISK | 171 | 176 | 157 | 176 | 174 | 188 | 173 | 171 | 184 | 
| ORB | BRISK | 73 | 74 | 79 | 85 | 79 | 92 | 90 | 88 | 91 |
| AKAZE | BRISK | 137 | 125 | 129 | 129 | 131 | 132 | 142 | 146 | 144 |
| SIFT | BRISK | 64 | 66 | 62 | 66 | 59 | 64 | 64 | 67 | 80 |
| SHITOMASI | SIFT | 112 | 109 | 104 | 103 | 99 | 101 | 96 | 106 | 97 |
| HARRIS | SIFT | 14 | 11 | 16 | 20 | 21 | 23 | 13 | 24 | 23 |
| FAST | SIFT | 316 | 325 | 297 | 311 | 291 | 326 | 315 | 300 | 301 |
| BRISK | SIFT | 182 | 193 | 169 | 183 | 171 | 195 | 194 | 176 | 183 |
| ORB | SIFT | 67 | 79 | 78 | 79 | 82 | 95 | 95 | 94 | 94 |
| AKAZE | SIFT | 134 | 134 | 130 | 136 | 137 | 147 | 147 | 154 | 151 |
| SIFT | SIFT | 82 | 81 | 85 | 93 | 90 | 81 | 82 | 102 | 104 |
| SHITOMASI | SHITOMASI |  |  |  |  |  |  |  |  |  |
| HARRIS | SHITOMASI |  |  |  |  |  |  |  |
| FAST | SHITOMASI |  |  |  |  |  |  |  |
| BRISK | SHITOMASI |  |  |  |  |  |  |  |
| ORB | SHITOMASI |  |  |  |  |  |  |  |
| AKAZE | SHITOMASI |  |  |  |  |  |  |  |
| SIFT | SHITOMASI |  |  |  |  |  |  |  |
| SHITOMASI | ORB |  |  |  |  |  |  |  |
| HARRIS | ORB |  |  |  |  |  |  |  |
| FAST | ORB |  |  |  |  |  |  |  |
| BRISK | ORB |  |  |  |  |  |  |  |
| ORB | ORB |  |  |  |  |  |  |  |
| AKAZE | ORB |  |  |  |  |  |  |  |
| SIFT | ORB |  |  |  |  |  |  |  |
| SHITOMASI | HARRIS |  |  |  |  |  |  |  | 
| HARRIS | HARRIS |  |  |  |  |  |  |  |
| FAST | HARRIS |  |  |  |  |  |  |  |
| BRISK | HARRIS |  |  |  |  |  |  |  |
| ORB | HARRIS |  |  |  |  |  |  |  |
| AKAZE | HARRIS |  |  |  |  |  |  |  |
| SIFT | HARRIS |  |  |  |  |  |  |  |
