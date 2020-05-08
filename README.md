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
Using following, any of the detectors can be used by selcting 'detectorType' 

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
Implementing the K-Nearest-Neighbor matching to check the descriptor distance ratio test.
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
            // The returned knn_matches vector contains some nested vectors with size less than 2
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

From above table, it is clearly evident that the FAST developed highest number of keypoint detections whereas HARRIS developed lowest number of keypoint detections for given set of images.

## MP.8 Performance Evaluation 2
Count the number of matched keypoints for all 10 images using all possible combinations of detectors and descriptors. In the matching step, the BF approach is used with the descriptor distance ratio set to 0.8.

NMP stands for Number of Matching Points

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
| FAST | BRISK | 2.27885 | 2.52872 | 4.80757 |
| FAST | BRISK | 2.10689 | 2.43084 | 4.53773 |
| FAST | BRISK | 2.16128 | 2.45608 | 4.61736 |
| FAST | BRISK | 2.16094 | 2.23948 | 4.40043 |
| FAST | BRISK | 2.17755 | 2.31374 | 4.49129 |
| FAST | BRISK | 2.0353 | 2.40661 | 4.44191 |
| FAST | BRISK | 2.15725 | 2.30369 | 4.46093 |
| FAST | BRISK | 2.21179 | 2.39982 | 4.61161 |
| FAST | BRISK | 2.14494 | 2.2666 | 4.41154 |
| FAST | SIFT | 2.35532 | 5.01408 | 7.3694 |
| FAST | SIFT | 2.08157 | 4.49641 | 6.57799 |
| FAST | SIFT | 2.12329 | 4.58486 | 6.70815 |
| FAST | SIFT | 2.02017 | 3.91231 | 5.93248 |
| FAST | SIFT | 2.19452 | 4.27686 | 6.47139 |
| FAST | SIFT | 2.20337 | 4.11131 | 6.31468 |
| FAST | SIFT | 2.18671 | 4.27912 | 6.46583 |
| FAST | SIFT | 2.22601 | 3.37203 | 5.59805 |
| FAST | SIFT | 2.1528 | 4.31104 | 6.46384 |
| FAST | ORB | 2.18843 | 3.09992 | 5.28835 |
| FAST | ORB | 2.13253 | 2.29299 | 4.42552 |
| FAST | ORB | 2.10614 | 2.24486 | 4.35101 |
| FAST | ORB | 2.10108 | 2.19642 | 4.29749 |
| FAST | ORB | 2.12479 | 2.11262 | 4.23742 |
| FAST | ORB | 2.07301 | 2.23336 | 4.30637 |
| FAST | ORB | 2.04615 | 2.19542 | 4.24156 |
| FAST | ORB | 2.15183 | 2.11012 | 4.26195 |
| FAST | ORB | 2.11646 | 2.09585 | 4.21232 |
| FAST | FREAK | 2.23726 | 2.48811 | 4.72537 |  
| FAST | FREAK | 2.07794 | 2.31345 | 4.39139 |
| FAST | FREAK | 2.29824 | 2.4314 | 4.72964 |
| FAST | FREAK | 2.06964 | 2.30565 | 4.37529 |
| FAST | FREAK | 2.15681 | 2.94984 | 5.10665 |
| FAST | FREAK | 2.18112 | 2.37242 | 4.55354 |
| FAST | FREAK | 2.07899 | 2.47994 | 4.55354 |
| FAST | FREAK | 2.14428 | 2.21617 | 4.36046 |
| FAST | FREAK | 2.1597 | 2.17128 | 4.33098 |
| FAST | BRIEF | 2.14541 | 2.93537 | 5.08078 |
| FAST | BRIEF | 2.1042 | 2.30366 | 4.40785 |
| FAST | BRIEF | 2.06488 | 2.2167 | 4.28158 |
| FAST | BRIEF | 2.11795 | 2.14099 | 4.25894 |
| FAST | BRIEF | 2.17487 | 2.11919 | 4.29406 |
| FAST | BRIEF | 2.05503 | 2.19162 | 4.24665 |
| FAST | BRIEF | 2.12311 | 2.17597 | 4.29908 |
| FAST | BRIEF | 2.15707 | 2.05393 | 4.211 |
| FAST | BRIEF | 2.15446 | 2.03347 | 4.18793 |
| BRISK | BRISK | 384.096 | 1.29134 | 385.387 | 
| BRISK | BRISK | 386.983 | 1.33444 | 388.317 | 
| BRISK | BRISK | 383.294 | 1.163225 | 384.926 | 
| BRISK | BRISK | 383.473 | 1.3166 | 384.79 | 
| BRISK | BRISK | 372.518 | 1.41092 | 373.929 | 
| BRISK | BRISK | 381.358 | 1.29837 | 382.656 | 
| BRISK | BRISK | 378.297 | 1.40106 | 379.698 | 
| BRISK | BRISK | 376.703 | 1.21853 | 377.921 | 
| BRISK | BRISK | 378.354 | 1.1049 | 379.459 | 
| BRISK | SIFT | 381.75 | 3.25568 | 385.006 |
| BRISK | SIFT | 384.107 | 2.67564 | 386.783 |
| BRISK | SIFT | 383.579 | 2.83622 | 386.415 |
| BRISK | SIFT | 384.064 | 2.29683 | 386.361 |
| BRISK | SIFT | 385.964 | 3.11723 | 389.081 |
| BRISK | SIFT | 383.961 | 3.02547 | 386.987 |
| BRISK | SIFT | 390.412 | 2.81946 | 393.231 |
| BRISK | SIFT | 380.711 | 2.45809 | 383.169 |
| BRISK | SIFT | 383.294 | 2.64124 | 385.935 |
| BRISK | ORB | 368.406 | 1.226 | 369.632 |
| BRISK | ORB | 371.016 | 1.21729 | 372.234 |
| BRISK | ORB | 366.757 | 1.18557 | 367.943 |
| BRISK | ORB | 366.234 | 1.2038 | 367.438 |
| BRISK | ORB | 366.648 | 1.26402 | 367.912 |
| BRISK | ORB | 373.725 | 1.20245 | 374.928 |
| BRISK | ORB | 372.55 | 1.18792 | 373.738 |
| BRISK | ORB | 366.598 | 1.12173 | 367.72 |
| BRISK | ORB | 366.855 | 1.07629 | 367.931 |
| BRISK | FREAK | 373.788 | 1.18144 | 374.969 |
| BRISK | FREAK | 369.861 | 1.14974 | 371.011 |
| BRISK | FREAK | 369.952 | 1.16672 | 371.119 |
| BRISK | FREAK | 377.511 | 1.21863 | 378.73 |
| BRISK | FREAK | 373.133 | 1.14651 | 374.28 |
| BRISK | FREAK | 367.627 | 1.15313 | 368.78 |
| BRISK | FREAK | 367.722 | 1.13272 | 368.855 |
| BRISK | FREAK | 366.382 | 1.04738 | 367.43 |
| BRISK | FREAK | 369.323 | 0.99643 | 370.32 |
| BRISK | BRIEF | 375.161 | 1.19097 | 376.352 |
| BRISK | BRIEF | 375.428 | 1.18698 | 376.615 |
| BRISK | BRIEF | 374.164 | 1.29344 | 375.458 |
| BRISK | BRIEF | 369.025 | 1.27271 | 370.298 |
| BRISK | BRIEF | 369.731 | 1.41471 | 371.146 |
| BRISK | BRIEF | 376.034 | 1.28794 | 377.322 |
| BRISK | BRIEF | 371.353 | 1.21569 | 372.569 |
| BRISK | BRIEF | 372.611 | 1.15407 | 373.765 |
| BRISK | BRIEF | 368.826 | 1.11711 | 369.944 |
| ORB | BRISK | 8.27133 | 0.41059 | 8.68192 |
| ORB | BRISK | 8.64094 | 0.315269 | 8.9562 |
| ORB | BRISK | 7.31895 | 0.39266 | 7.71161 |
| ORB | BRISK | 7.53117 | 0.448899 | 7.98006 |
| ORB | BRISK | 7.45703 | 0.371798 | 7.82883 |
| ORB | BRISK | 7.7517 | 0.409414 | 8.16111 |
| ORB | BRISK | 8.42251 | 0.412985 | 8.83549 |
| ORB | BRISK | 7.76418 | 0.522109 | 8.28269 |
| ORB | BRISK | 8.59146 | 0.414037 | 9.0055 |
| ORB | SIFT | 7.57025 | 0.388453 | 7.9587 |
| ORB | SIFT | 8.23043 | 0.402443 | 8.63287 |
| ORB | SIFT | 7.6069 | 0.487275 | 8.09418 |
| ORB | SIFT | 7.97219 | 0.608296 | 8.58048 |
| ORB | SIFT | 7.44699 | 0.522933 | 7.96992 |
| ORB | SIFT | 7.82048 | 0.612691 | 8.43317 |
| ORB | SIFT | 7.95474 | 0.685999 | 8.64074 |
| ORB | SIFT | 7.95987 | 0.777506 | 8.73738 |
| ORB | SIFT | 7.98711 | 0.644184 | 8.63129 |
| ORB | ORB | 7.92491 | 3.78587 | 11.7108 |
| ORB | ORB | 8.14426 | 0.496772 | 8.64103 |
| ORB | ORB | 7.67519 | 0.347672 | 8.02286 |
| ORB | ORB | 7.38037 | 0.384949 | 7.76532 |
| ORB | ORB | 7.37435 | 0.383176 | 7.75752 |
| ORB | ORB | 7.68234 | 0.443866 | 8.12621 |
| ORB | ORB | 7.79095 | 0.41868 | 8.20963 |
| ORB | ORB | 7.36014 | 0.401922 | 7.76206 |
| ORB | ORB | 7.79095 | 0.451838 | 8.96641 |
| ORB | FREAK | 8.58662 | 0.279452 | 8.86607 |
| ORB | FREAK | 7.80272 | 0.211074 | 8.01379 |
| ORB | FREAK | 7.40833 | 0.313755 | 7.72208 |
| ORB | FREAK | 7.47014 | 0.219333 | 7.68947 |
| ORB | FREAK | 7.48631 | 0.240278 | 7.72659 |
| ORB | FREAK | 7.41132 | 0.260363 | 7.67168 |
| ORB | FREAK | 7.42458 | 0.251345 | 7.67593 |
| ORB | FREAK | 7.66387 | 0.257967 | 7.92184 |
| ORB | FREAK | 7.65352 | 0.267107 | 7.92063 |
| ORB | BRIEF | 8.77225 | 0.422916 | 9.19516 |
| ORB | BRIEF | 8.21333 | 0.318287 | 8.53162 |
| ORB | BRIEF | 7.421 | 0.326663 | 7.74766 |
| ORB | BRIEF | 7.48958 | 0.388515 | 7.8781 |
| ORB | BRIEF | 7.44991 |0.366591  | 7.81651 |
| ORB | BRIEF | 7.42039 | 0.392069 | 7.81246 |
| ORB | BRIEF | 7.83177 | 0.382792 | 8.21457 |
| ORB | BRIEF | 7.82049 | 0.382469 | 8.20296 |
| ORB | BRIEF | 8.552 | 0.387784 | 8.93978 |
| AKAZE | BRISK | 91.1065 | 0.621303 | 91.7278 |
| AKAZE | BRISK | 85.0008 | 0.713556 | 85.7144 |
| AKAZE | BRISK | 83.98 | 0.59437 | 84.5744 |
| AKAZE | BRISK | 91.4649 | 0.583762 | 92.0487 |
| AKAZE | BRISK | 90.9958 | 0.576727 | 91.5726 |
| AKAZE | BRISK | 90.2266 | 0.702051 | 90.9287 |
| AKAZE | BRISK | 85.1975 | 0.646576 | 85.844 |
| AKAZE | BRISK | 93.2309 | 0.779683 | 94.0106 |
| AKAZE | BRISK | 85.4779 | 0.678123 | 86.1561 |
| AKAZE | SIFT | 84.3524 | 1.01345 | 85.3658 |
| AKAZE | SIFT | 92.1947 | 0.958694 | 93.1534 |
| AKAZE | SIFT | 87.0436 | 0.695435 | 87.7391 |
| AKAZE | SIFT | 90.0693 | 0.854863 | 90.9241 |
| AKAZE | SIFT | 88.0069 | 0.729106 | 88.736 |
| AKAZE | SIFT | 80.0619 | 1.02719 | 81.0891 |
| AKAZE | SIFT | 80.4279 | 0.799904 | 81.2296 |
| AKAZE | SIFT | 79.0344 | 1.08956 | 80.124 |
| AKAZE | SIFT | 85.4055 | 0.881244 | 86.2867 |
| AKAZE | ORB | 87.7415 | 0.584345 | 88.3258 |
| AKAZE | ORB | 87.3345 | 0.558069 | 87.8926 |
| AKAZE | ORB | 82.9088 | 0.529649 | 83.4384 |
| AKAZE | ORB | 89.9356 | 0.517251 | 90.4528 |
| AKAZE | ORB | 85.0691 | 0.558913 | 85.628 |
| AKAZE | ORB | 82.1256 | 0.59063 | 82.7162 |
| AKAZE | ORB | 88.6616 | 0.593602 | 89.2552 |
| AKAZE | ORB | 90.6514 | 0.624375 | 91.2752 |
| AKAZE | ORB | 84.3908 | 0.650172 | 85.041 |
| AKAZE | FREAK | 85.4262 | 0.618818 | 86.0451 |
| AKAZE | FREAK | 89.4448 | 0.569168 | 90.014 |
| AKAZE | FREAK | 72.2262 | 0.575073 | 72.8012 |
| AKAZE | FREAK | 71.1157 | 0.58822 | 71.7039 |
| AKAZE | FREAK | 77.7198 | 0.637659 | 78.3574 |
| AKAZE | FREAK | 68.5675 | 0.604312 | 69.1718 |
| AKAZE | FREAK | 73.2304 | 0.713459 | 73.9438 |
| AKAZE | FREAK | 74.5598 | 0.647722 | 75.2075 |
| AKAZE | FREAK | 74.1285 | 0.669831 | 74.7983 |
| AKAZE | AKAZE | 74.7486 | 0.657888 | 75.4064 |
| AKAZE | AKAZE | 84.2437 | 0.600763 | 84.8445 |
| AKAZE | AKAZE | 82.7939 | 0.570724 | 83.3646 |
| AKAZE | AKAZE | 78.1789 | 0.578843 | 78.7578 |
| AKAZE | AKAZE | 87.6659 | 0.664232 | 88.3302 |
| AKAZE | AKAZE | 83.4093 | 0.663689 | 84.073 |
| AKAZE | AKAZE | 82.5089 | 0.672959 | 83.1819 |
| AKAZE | AKAZE | 78.6007 | 0.679143 | 79.2799 |
| AKAZE | AKAZE | 80.7148 | 0.717357 | 81.4321 |
| AKAZE | BRIEF | 92.2412 | 0.604102 | 92.8453 |
| AKAZE | BRIEF | 84.8899 | 0.515478 | 85.4054 |
| AKAZE | BRIEF | 90.1345 | 0.536594 | 90.671 |
| AKAZE | BRIEF | 89.6698 | 0.545557 | 90.2154 |
| AKAZE | BRIEF | 84.1295 | 0.539823 | 84.6694 |
| AKAZE | BRIEF | 83.7279 | 0.544556 | 84.2724 |
| AKAZE | BRIEF | 83.3214 | 0.583044 | 83.9044 |
| AKAZE | BRIEF | 84.9991 | 0.587727 | 85.5868 |
| AKAZE | BRIEF | 87.881 | 0.631628 | 88.5126 |
| SIFT | BRISK | 135.121 | 0.532099 | 135.653 |
| SIFT | BRISK | 126.589 | 0.481522 | 127.071 |
| SIFT | BRISK | 103.009 | 0.506925 | 103.515 |
| SIFT | BRISK | 101.998 | 0.468628 | 102.467 |
| SIFT | BRISK | 104.917 | 0.476771 | 105.394 |
| SIFT | BRISK | 102.029 | 0.512564 | 102.541 |
| SIFT | BRISK | 100.16 | 0.503186 | 100.663 |
| SIFT | BRISK | 104.278 | 0.568011 | 104.846 |
| SIFT | BRISK | 104.237 | 0.538255 | 104.775 |
| SIFT | SIFT | 99.6419 | 0.676649 | 100.319 |
| SIFT | SIFT | 115.241 | 0.576869 | 115.818 |
| SIFT | SIFT | 94.8269 | 0.553969 | 95.3809 |
| SIFT | SIFT | 96.6686 | 0.546637 | 97.2152 |
| SIFT | SIFT | 102.924 | 0.55249 | 103.476 |
| SIFT | SIFT | 99.8496 | 0.618809 | 100.468 |
| SIFT | SIFT | 99.6486 | 0.596442 | 100.245 |
| SIFT | SIFT | 98.2274 | 0.715122 | 98.9425 |
| SIFT | SIFT | 101.132 | 0.71257 | 101.845 |
| SIFT | FREAK | 135.5 | 0.560221 | 136.06 |
| SIFT | FREAK | 135.98 | 0.440238 | 136.421 |
| SIFT | FREAK | 128.081 | 0.439205 | 128.521 |
| SIFT | FREAK | 127.934 | 0.4579 | 128.392 |
| SIFT | FREAK | 127.608 | 0.453994 | 128.062 |
| SIFT | FREAK | 132.987 | 0.475086 | 133.463 |
| SIFT | FREAK | 127.062 | 0.479452 | 127.542 |
| SIFT | FREAK | 128.472 | 0.530203 | 129.002 |
| SIFT | FREAK | 128.558 | 0.554995 | 129.113 |
| SIFT | BRIEF | 142.513 | 0.527748 | 143.04 |
| SIFT | BRIEF | 133.356 | 0.43725 | 133.973 |
| SIFT | BRIEF | 135.841 | 0.446556 | 136.287 |
| SIFT | BRIEF | 132.896 | 0.437383 | 133.333 |
| SIFT | BRIEF | 136.033 | 0.437899 | 136.471 |
| SIFT | BRIEF | 138.873 | 0.482608 | 139.355 |
| SIFT | BRIEF | 139.131 | 0.448676 | 139.58 |
| SIFT | BRIEF | 136.731 | 0.526625 | 137.258 |
| SIFT | BRIEF | 133.428 | 0.500759 | 133.929 |

From above tables (MP.8 and MP.9), the TOP 3 detector/descriptor combinations are as follows:
1. FAST detectors and BRIEF descriptors
2. FAST detectors and ORB descriptors
3. FAST detectors and SIFT descriptors

It is strongly evident that the number of matching keypoints with FAST detectors is around 300+ for its combination with either of the three descriptors (BRIEF,ORB and SIFT). Also, the time required for keypoint detections and descriptors extraction is less than 8 ms for all the 3 combinations. Thus any of the above 3 combination is recommended as best choice for the purpose of detecting keypoints on vehicles.
