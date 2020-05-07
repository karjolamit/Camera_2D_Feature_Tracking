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
| SHITOMASI | SIFT | 112 | 109 | 104 | 103 | 99 | 101 | 96 | 106 | 97 |
| SHITOMASI | ORB | 106 | 102 | 99 | 102 | 103 | 91 | 98 | 104 | 97 |
| SHITOMASI | FREAK | 86 | 90 | 86 | 88 | 86 | 80 | 81 | 86 | 85 |
| SHITOMASI | BRIEF | 115 | 111 | 104 | 101 | 102 | 102 | 100 | 109 | 100 |
| HARRIS | BRISK | 12 | 10 | 14 | 16 | 16 | 17 | 15 | 22 | 21 |
| HARRIS | SIFT | 14 | 11 | 16 | 20 | 21 | 23 | 13 | 24 | 23 |
| HARRIS | ORB | 12 | 12 | 15 | 19 | 23 | 21 | 15 | 25 | 23 |
| HARRIS | FREAK | 13 | 12 | 15 | 16 | 16 | 21 | 12 | 21 | 19 |
| HARRIS | BRIEF | 14 | 11 | 16 | 21 | 23 | 28 | 16 | 25 | 24 |
| FAST | BRISK | 256 | 243 | 241 | 239 | 215 | 251 | 248 | 243 | 247 |
| FAST | SIFT | 316 | 325 | 297 | 311 | 291 | 326 | 315 | 300 | 301 |
| FAST | ORB | 307 | 308 | 298 | 321 | 283 | 315 | 323 | 302 | 311 |
| FAST | FREAK | 251 | 247 | 233 | 255 | 231 | 265 | 251 | 253 | 247 |
| FAST | BRIEF | 320 | 332 | 299 | 331 | 276 | 327 | 324 | 315 | 307 |
| BRISK | BRISK | 171 | 176 | 157 | 176 | 174 | 188 | 173 | 171 | 184 | 
| BRISK | SIFT | 182 | 193 | 169 | 183 | 171 | 195 | 194 | 176 | 183 |
| BRISK | ORB | 162 | 175 | 158 | 167 | 160 | 182 | 167 | 171 | 172 |
| BRISK | FREAK | 160 | 177 | 155 | 173 | 161 | 183 | 169 | 178 | 168 |
| BRISK | BRIEF | 178 | 205 | 185 | 179 | 183 | 195 | 207 | 189 | 183 |
| ORB | BRISK | 73 | 74 | 79 | 85 | 79 | 92 | 90 | 88 | 91 |
| ORB | SIFT | 67 | 79 | 78 | 79 | 82 | 95 | 95 | 94 | 94 |
| ORB | ORB | 67 | 70 | 72 | 84 | 91 | 101 | 92 | 93 | 93 |
| ORB | FREAK | 42 | 36 | 44 | 47 | 44 | 51 | 52 | 48 | 56 |
| ORB | BRIEF | 49 | 43 | 45 | 59 | 53 | 78 | 68 | 84 | 66 |
| AKAZE | BRISK | 137 | 125 | 129 | 129 | 131 | 132 | 142 | 146 | 144 |
| AKAZE | SIFT | 134 | 134 | 130 | 136 | 137 | 147 | 147 | 154 | 151 |
| AKAZE | ORB | 131 | 129 | 127 | 117 | 130 | 131 | 137 | 135 | 145 |
| AKAZE | FREAK | 126 | 129 | 127 | 121 | 122 | 133 | 144 | 147 | 138 |
| AKAZE | AKAZE | 138 | 138 | 133 | 127 | 129 | 146 | 147 | 151 | 150 |
| AKAZE | BRIEF | 141 | 134 | 131 | 130 | 134 | 146 | 150 | 148 | 152 |
| SIFT | BRISK | 64 | 66 | 62 | 66 | 59 | 64 | 64 | 67 | 80 |
| SIFT | SIFT | 82 | 81 | 85 | 93 | 90 | 81 | 82 | 102 | 104 |
| SIFT | FREAK | 65 | 72 | 64 | 66 | 59 | 59 | 64 | 65 | 79 |
| SIFT | BRIEF | 86 | 78 | 76 | 85 | 69 | 74 | 76 | 70 | 88 |

## MP.9 Performance Evaluation 3
Time required for keypoint detection and descriptor extraction. 

| Detector Type | Descriptor Type | Detection Time | Description Time | Total Time |
| ------------- | --------------- | -------------- | ---------------- | ---------- | 
| SHITOMASI | BRISK | 18.0882 | 0.558094 | 18.6463 |  
| SHITOMASI | BRISK | 18.7076 | 0.481847 | 19.1894 |  
| SHITOMASI | BRISK | 17.8675 | 0.480884 | 18.3484 | 
| SHITOMASI | BRISK | 16.7682 | 0.461885 | 17.2301 |  
| SHITOMASI | BRISK | 17.6994 | 0.451701 | 18.1511 | 
| SHITOMASI | BRISK | 17.6489 | 0.461644 | 18.1105 | 
| SHITOMASI | BRISK | 17.0999 | 0.467073 | 17.567 | 
| SHITOMASI | BRISK | 16.839 | 0.45153 | 17.2905 | 
| SHITOMASI | BRISK | 16.6246 | 0.377672 | 17.0022 | 
| SHITOMASI | SIFT | 11.7499 | 0.731261 | 12.4812 | 
| SHITOMASI | SIFT | 16.833 | 0.528927 | 17.3619 | 
| SHITOMASI | SIFT | 17.1126 | 0.522772 | 17.6353 | 
| SHITOMASI | SIFT | 14.4156 | 0.469529 | 14.8851 | 
| SHITOMASI | SIFT | 12.2233 | 0.630024 | 12.8534 | 
| SHITOMASI | SIFT | 16.7057 | 0.484108 | 17.1898 | 
| SHITOMASI | SIFT | 15.7316 | 0.470592 | 16.2021 | 
| SHITOMASI | SIFT | 12.0244 | 0.502133 | 12.5266 | 
| SHITOMASI | SIFT | 15.5991 | 0.422178 | 16.0213 | 
| SHITOMASI | ORB | 21.5529 | 0.459044 | 22.012 |
| SHITOMASI | ORB | 18.0554 | 0.374417 | 18.4298 |
| SHITOMASI | ORB | 17.054 | 0.399398 | 17.4534 |
| SHITOMASI | ORB | 16.7836 | 0.459618 | 17.2432 |
| SHITOMASI | ORB | 15.5779 | 0.362314 | 15.9403 |
| SHITOMASI | ORB | 16.8071 | 0.427296 | 17.2344 |
| SHITOMASI | ORB | 17.0359 | 0.368791 | 17.4047 |
| SHITOMASI | ORB | 16.5694 | 0.382373 | 16.9518 |
| SHITOMASI | ORB | 15.7252 | 0.350452 | 16.0756 |
| SHITOMASI | FREAK | 16.8901 | 0.527649 | 17.4177 |
| SHITOMASI | FREAK | 17.2911 | 0.473072 | 17.7642 |
| SHITOMASI | FREAK | 12.7285 | 0.471825 | 13.2003 |
| SHITOMASI | FREAK | 11.9275 | 0.472993 | 12.4005 |
| SHITOMASI | FREAK | 12.0041 | 0.452552 | 12.4566 |
| SHITOMASI | FREAK | 12.0593 | 0.387981 | 12.4473 |
| SHITOMASI | FREAK | 12.0784 | 0.401531 | 12.4799 |
| SHITOMASI | FREAK | 11.8781 | 0.388446 | 12.2665 |
| SHITOMASI | FREAK | 11.838 | 0.435893 | 12.2739 |
| SHITOMASI | BRIEF | 17.6987 | 7.36804 | 25.0667 |
| SHITOMASI | BRIEF | 18.3944 | 0.397624 | 18.7921 |
| SHITOMASI | BRIEF | 17.9506 | 0.368199 | 18.3188 |
| SHITOMASI | BRIEF | 17.2261 | 0.404574 | 17.6307 |
| SHITOMASI | BRIEF | 15.6312 | 0.359683 | 15.9908 |
| SHITOMASI | BRIEF | 16.5427 | 0.355996 | 16.8987 |
| SHITOMASI | BRIEF | 17.3645 | 0.371467 | 17.7359 |
| SHITOMASI | BRIEF | 16.9688 | 0.371575 | 17.3403 |
| SHITOMASI | BRIEF | 16.7046 | 0.345939 | 17.0505 |
| HARRIS | BRISK | 18.2993 | 0.23954 | 18.5389 |
| HARRIS | BRISK | 19.1793 | 0.085311 | 19.2646 |
| HARRIS | BRISK | 18.6764 | 0.150347 | 18.8267 |
| HARRIS | BRISK | 18.3236 | 0.186497 | 18.5101 |
| HARRIS | BRISK | 38.5627 | 0.176797 | 38.7395 |
| HARRIS | BRISK | 17.3974 | 0.199732 |  17.5971|
| HARRIS | BRISK | 20.6418 | 0.12516 | 20.767 |
| HARRIS | BRISK | 18.4205 | 0.168315 | 18.5888 |
| HARRIS | BRISK | 23.5975 | 0.207698 | 23.8051 |
| HARRIS | SIFT | 17.5876 | 0.170662 | 17.7583 |
| HARRIS | SIFT | 19.3157 | 0.126354 | 19.442 |
| HARRIS | SIFT | 19.3777 | 0.088685 | 19.4663 |
| HARRIS | SIFT | 18.8119 | 0.100681 | 18.9126 |
| HARRIS | SIFT | 38.3559 | 0.135887 | 38.4917 |
| HARRIS | SIFT | 16.9305 | 0.151526 | 17.082 |
| HARRIS | SIFT | 20.5608 | 0.176257 | 20.737 |
| HARRIS | SIFT | 19.3551 | 0.148628 | 19.5037 |
| HARRIS | SIFT | 24.7814 | 0.157323 | 24.9387 |
| HARRIS | ORB | 17.7508 | 0.171317 | 17.9221 | 
| HARRIS | ORB | 18.54 | 0.0951 | 18.6351 | 
| HARRIS | ORB | 18.1554 | 0.101276 | 18.2567 | 
| HARRIS | ORB | 18.1972 | 0.142282 | 18.3395 | 
| HARRIS | ORB | 37.7491 | 0.170737 | 37.9198 | 
| HARRIS | ORB | 16.8466 | 0.167801 | 17.0144 | 
| HARRIS | ORB | 21.1824 | 0.182322 | 21.3647 | 
| HARRIS | ORB | 18.1493 | 0.17787 | 18.3272 | 
| HARRIS | ORB | 23.4496 | 0.145925 | 23.5955 | 
| HARRIS | FREAK | 17.3175 | 0.177895 | 17.4954 |
| HARRIS | FREAK | 17.1821 | 0.110356 | 17.2925 |
| HARRIS | FREAK | 13.33 | 0.13726 | 13.4673 |
| HARRIS | FREAK | 13.2134 | 0.189617 | 13.4031 |
| HARRIS | FREAK | 33.6382 | 0.30832 | 33.9465 |
| HARRIS | FREAK | 12.4656 | 0.150676 | 12.6162 |
| HARRIS | FREAK | 14.9118 | 0.148541 | 15.0603 |
| HARRIS | FREAK | 13.5546 | 0.208598 | 13.7632 |
| HARRIS | FREAK | 19.5708 | 0.187201 | 19.758 |
| HARRIS | BRIEF | 18.114 | 0.141947 | 18.256 |
| HARRIS | BRIEF | 18.783 | 0.077317 | 18.8603 |
| HARRIS | BRIEF | 17.9862 | 0.12472 | 18.111 |
| HARRIS | BRIEF | 18.3752 | 0.107121 | 18.4823 |
| HARRIS | BRIEF | 37.7216 | 0.182259 | 37.9038 |
| HARRIS | BRIEF | 16.8995 | 0.121237 | 17.0207 |
| HARRIS | BRIEF | 20.5169 | 0.120932 | 20.6379 |
| HARRIS | BRIEF | 18.1029 | 0.145753 | 18.2486 |
| HARRIS | BRIEF | 23.5871 | 0.159954 | 23.747 |
| FAST | BRISK |  |  |  |
| FAST | BRISK |  |  |  |
| FAST | BRISK |  |  |  |
| FAST | BRISK |  |  |  |
| FAST | BRISK |  |  |  |
| FAST | BRISK |  |  |  |
| FAST | BRISK |  |  |  |
| FAST | BRISK |  |  |  |
| FAST | BRISK |  |  |  |
| FAST | SIFT |  |  |  |
| FAST | SIFT |  |  |  |
| FAST | SIFT |  |  |  |
| FAST | SIFT |  |  |  |
| FAST | SIFT |  |  |  |
| FAST | SIFT |  |  |  |
| FAST | SIFT |  |  |  |
| FAST | SIFT |  |  |  |
| FAST | SIFT |  |  |  |
| FAST | ORB |  |  |  |
| FAST | ORB |  |  |  |
| FAST | ORB |  |  |  |
| FAST | ORB |  |  |  |
| FAST | ORB |  |  |  |
| FAST | ORB |  |  |  |
| FAST | ORB |  |  |  |
| FAST | ORB |  |  |  |
| FAST | ORB |  |  |  |
| FAST | FREAK |  |  |  
| FAST | FREAK |  |  |  |
| FAST | FREAK |  |  |  |
| FAST | FREAK |  |  |  |
| FAST | FREAK |  |  |  |
| FAST | FREAK |  |  |  |
| FAST | FREAK |  |  |  |
| FAST | FREAK |  |  |  |
| FAST | FREAK |  |  |  |
| FAST | BRIEF |  |  |  |
| FAST | BRIEF |  |  |  |
| FAST | BRIEF |  |  |  |
| FAST | BRIEF |  |  |  |
| FAST | BRIEF |  |  |  |
| FAST | BRIEF |  |  |  |
| FAST | BRIEF |  |  |  |
| FAST | BRIEF |  |  |  |
| FAST | BRIEF |  |  |  |
| BRISK | BRISK |  |  |  | 
| BRISK | BRISK |  |  |  | 
| BRISK | BRISK |  |  |  | 
| BRISK | BRISK |  |  |  | 
| BRISK | BRISK |  |  |  | 
| BRISK | BRISK |  |  |  | 
| BRISK | BRISK |  |  |  | 
| BRISK | BRISK |  |  |  | 
| BRISK | BRISK |  |  |  | 
| BRISK | SIFT |  |  |  |
| BRISK | SIFT |  |  |  |
| BRISK | SIFT |  |  |  |
| BRISK | SIFT |  |  |  |
| BRISK | SIFT |  |  |  |
| BRISK | SIFT |  |  |  |
| BRISK | SIFT |  |  |  |
| BRISK | SIFT |  |  |  |
| BRISK | SIFT |  |  |  |
| BRISK | ORB |  |  |  |
| BRISK | ORB |  |  |  |
| BRISK | ORB |  |  |  |
| BRISK | ORB |  |  |  |
| BRISK | ORB |  |  |  |
| BRISK | ORB |  |  |  |
| BRISK | ORB |  |  |  |
| BRISK | ORB |  |  |  |
| BRISK | ORB |  |  |  |
| BRISK | FREAK |  |  |  |
| BRISK | FREAK |  |  |  |
| BRISK | FREAK |  |  |  |
| BRISK | FREAK |  |  |  |
| BRISK | FREAK |  |  |  |
| BRISK | FREAK |  |  |  |
| BRISK | FREAK |  |  |  |
| BRISK | FREAK |  |  |  |
| BRISK | FREAK |  |  |  |
| BRISK | BRIEF |  |  |  |
| BRISK | BRIEF |  |  |  |
| BRISK | BRIEF |  |  |  |
| BRISK | BRIEF |  |  |  |
| BRISK | BRIEF |  |  |  |
| BRISK | BRIEF |  |  |  |
| BRISK | BRIEF |  |  |  |
| BRISK | BRIEF |  |  |  |
| BRISK | BRIEF |  |  |  |
| ORB | BRISK |  |  |  |
| ORB | BRISK |  |  |  |
| ORB | BRISK |  |  |  |
| ORB | BRISK |  |  |  |
| ORB | BRISK |  |  |  |
| ORB | BRISK |  |  |  |
| ORB | BRISK |  |  |  |
| ORB | BRISK |  |  |  |
| ORB | BRISK |  |  |  |
| ORB | SIFT |  |  |  |
| ORB | SIFT |  |  |  |
| ORB | SIFT |  |  |  |
| ORB | SIFT |  |  |  |
| ORB | SIFT |  |  |  |
| ORB | SIFT |  |  |  |
| ORB | SIFT |  |  |  |
| ORB | SIFT |  |  |  |
| ORB | SIFT |  |  |  |
| ORB | ORB |  |  |  |
| ORB | ORB |  |  |  |
| ORB | ORB |  |  |  |
| ORB | ORB |  |  |  |
| ORB | ORB |  |  |  |
| ORB | ORB |  |  |  |
| ORB | ORB |  |  |  |
| ORB | ORB |  |  |  |
| ORB | ORB |  |  |  |
| ORB | FREAK |  |  |  |
| ORB | FREAK |  |  |  |
| ORB | FREAK |  |  |  |
| ORB | FREAK |  |  |  |
| ORB | FREAK |  |  |  |
| ORB | FREAK |  |  |  |
| ORB | FREAK |  |  |  |
| ORB | FREAK |  |  |  |
| ORB | FREAK |  |  |  |
| ORB | BRIEF |  |  |  |
| ORB | BRIEF |  |  |  |
| ORB | BRIEF |  |  |  |
| ORB | BRIEF |  |  |  |
| ORB | BRIEF |  |  |  |
| ORB | BRIEF |  |  |  |
| ORB | BRIEF |  |  |  |
| ORB | BRIEF |  |  |  |
| ORB | BRIEF |  |  |  |
| AKAZE | BRISK |  |  |  |
| AKAZE | SIFT |  |  |  |
| AKAZE | ORB |  |  |  |
| AKAZE | FREAK |  |  |  |
| AKAZE | AKAZE |  |  |  |
| AKAZE | BRIEF |  |  |  |
| SIFT | BRISK |  |  |  |
| SIFT | SIFT |  |  |  |
| SIFT | FREAK |  |  |  |
| SIFT | BRIEF |  |  |  |


the TOP3 detector / descriptor combinations must be recommended as the best choice for our purpose of detecting keypoints on vehicles.
