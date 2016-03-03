#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/stitching/detail/blenders.hpp>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <limits>

using namespace cv;
using namespace std;

// Functions prototypes
void Homography(vector<Mat> Images, vector<Mat> transforms);
void FindOutputLimits(vector<Mat> Images, vector<Mat> transforms, int xMin, int xMax, int yMin, int yMax);
//void warpMasks(…);
//void warpImages(…);
//void BlendImages(…);

int main()
{
    // Initialize OpenCV nonfree module
    initModule_nonfree();
    // Set the dir/name of each image
    const int NUM_IMAGES = 2; // 6 for the real panorama
    const string basedir = "C:\\Users\\Lazy\\Documents\\GitHub\\Computer_Vision\\Assignment3\\images";
    const string IMG_NAMES[] = { basedir + "\\Img1.jpg", basedir + "\\Img2.jpg" };

    // Load the images
    vector<Mat> Images;
    for (int i = 0; i < NUM_IMAGES; i++) {
        Mat img = imread(IMG_NAMES[i], CV_LOAD_IMAGE_COLOR);
        // check image data
        if (!img.data) {
            cout << "\t\tError loading image in " << IMG_NAMES[i] << endl;
            return -1;
        }

        Images.push_back(img);

        // display the image set
        imshow(IMG_NAMES[i], Images[i]);
        waitKey(0);
        destroyWindow(IMG_NAMES[i]);
    }

    cout << "Loaded images" << endl;

    // 1. Initialize all the transforms to the identity matrix
    vector<Mat> transforms;
    for (int i = 0; i < NUM_IMAGES; i++) {
        transforms.push_back(Mat::eye(3, 3, CV_64F));
    }

    // 2. Calculate the transformation matrices
    Homography(Images, transforms);

    // 3. Compute the min and max limits of the transformations
    int xMin, xMax, yMin, yMax;
    FindOutputLimits(Images, transforms, xMin, xMax, yMin, yMax);

    // 4. Compute the size of the panorama
    int width = xMax - xMin + 1;
    int height = yMax - yMin + 1;

    cout << xMax << ", " << xMin << ", " << yMax << ", " << yMin << endl;
    cout << width << ", " << height << endl;

    // 5. Initialize the panorama image
    Mat panorama = Mat(height, width, CV_64F);

    // 6. Initialize warped mask images

    // 7. Warp the mask images
//    warpMasks(Images, masks_warped, transforms, panorama);

    // 8. Warp the images
//    warpImages(Images, masks_warped, transforms, panorama);

    // 9. Initialize the blended panorama images

    // 10. Blending
//    BlendImages(Images, pano_feather, pano_multiband, masks_warped, transforms);

    return 0;
}

// match features in images
void Homography(vector<Mat> Images, vector<Mat> transforms) {

    // Create a SIFT feature detector object
    Ptr<FeatureDetector> featureDetector = FeatureDetector::create("SIFT");

    // Create a SIFT descriptor extractor object
    Ptr<DescriptorExtractor> descriptorExtractor = DescriptorExtractor::create("SIFT");

    // starting from second image, detect features and match
    for (unsigned int i = 1; i < Images.size(); i ++) {
        BFMatcher matcher;

        // detect key points
        vector<KeyPoint> keyPointsPrev;
        vector<KeyPoint> keyPointsCurr;
        featureDetector->detect(Images[i-1], keyPointsPrev);
        featureDetector->detect(Images[i], keyPointsCurr);

        // compute SIFT descriptors
        Mat descriptorPrev;
        Mat descriptorCurr;
        descriptorExtractor->compute(Images[i-1], keyPointsPrev, descriptorPrev);
        descriptorExtractor->compute(Images[i], keyPointsCurr, descriptorCurr);

        // match the descriptors
        vector<DMatch> matchResults;
        matcher.match(descriptorPrev, descriptorCurr, matchResults);

        // display the matches
        Mat outIm;
        drawMatches(Images[i-1], keyPointsPrev, Images[i], keyPointsCurr, matchResults, outIm, DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
        stringstream ss;
        ss << "Matched Images: " << i-1 << ", " << i;
        imshow(ss.str(), outIm);
        waitKey(0);
        destroyWindow(ss.str());

        vector<Point2d> matchedPointsPrev;
        vector<Point2d> matchedPointsCurr;
        for (unsigned int i = 0; i < matchResults.size(); i++) {
            matchedPointsPrev.push_back(Point2d(keyPointsPrev[i].pt.x, keyPointsPrev[i].pt.y));
            matchedPointsCurr.push_back(Point2d(keyPointsCurr[i].pt.x, keyPointsCurr[i].pt.y));
        }

        // estimate transform between images i and i-1
        Mat T = findHomography(matchedPointsCurr, matchedPointsPrev, CV_RANSAC);

        // compute transform to the ref
        transforms[i] = T * transforms[i-1];
    }
}

void FindOutputLimits(vector<Mat> Images, vector<Mat> transforms, int xMin, int xMax, int yMin, int yMax) {
    // get the corners of the first image
    int height = Images[0].size().height;
    int width = Images[0].size().width;

    // create the corners
    vector<Mat> corners;
    Mat corner = Mat::zeros(3, 1, CV_64F);
    corner.at<float>(2, 0) = 1;
    corners.push_back(corner);
    corner.at<float>(1, 0) = height - 1;
    corners.push_back(corner);
    corner.at<float>(0,0) = width - 1;
    corners.push_back(corner);
    corner.at<float>(1,0) = 0;
    corners.push_back(corner);

    cout << "Finding min and max corners" << endl;

    // project and find the min and max
    xMin = numeric_limits<int>::max();
    xMax = numeric_limits<int>::min();
    yMin = numeric_limits<int>::max();
    yMax = numeric_limits<int>::min();
    for (unsigned int i = 0; i < transforms.size(); i++) {
        for (unsigned int j = 0; j < corners.size(); j++) {
            Mat projected = transforms[i] * corners[j];
            if (projected.at<float>(0,0) < xMin && projected.at<float>(1,0) < yMin) {
                xMin = projected.at<float>(0,0);
                yMin = projected.at<float>(1,0);
            }
            if (projected.at<float>(0,0) > xMax && projected.at<float>(1,0) > yMax) {
                xMax = projected.at<float>(0,0);
                yMax = projected.at<float>(1,0);
            }
        }
    }

    cout << "transforms" << endl;

    // translate all images so that xMin and yMin become zero
    Mat translation = Mat::eye(3, 3, CV_64F);
    translation.at<float>(0,2) = -1 * xMin;
    translation.at<float>(1,2) = -1 * yMin;

    for (unsigned int i = 0; i < transforms.size(); i++) {
        transforms[i] = translation * transforms[i];
    }
}

//void warpMasks(…) {

//}

//void warpImages(…) {

//}

//void BlendImages(…) {

//}
