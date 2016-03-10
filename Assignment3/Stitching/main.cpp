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
void Homography(vector<Mat> Images, vector<Mat> &transforms);
void FindOutputLimits(vector<Mat> Images, vector<Mat> &transforms, int &xMin, int &xMax, int &yMin, int &yMax);
void warpMasks(vector<Mat> Images, vector<Mat> &masks_warped, vector<Mat> transforms, Mat &panorama);
void warpImages(vector<Mat> Images, vector<Mat> &masks_warped, vector<Mat> transforms, Mat &panorama);
void BlendImages(vector<Mat> Images, Mat &pano_feather, Mat &pano_multiband, vector<Mat> masks_warped, vector<Mat> transforms, Size pano_size);

int main()
{
    // Initialize OpenCV nonfree module
    initModule_nonfree();

    // Set the dir/name of each image
    const int NUM_IMAGES = 6; // 6 for the real panorama
    const string basedir = "C:\\Users\\Lazy\\Documents\\GitHub\\Computer_Vision\\Assignment3\\images";
    const string IMG_NAMES[] = { basedir + "\\Img6.jpg", basedir + "\\Img5.jpg", basedir + "\\Img4.jpg", basedir + "\\Img3.jpg", basedir + "\\Img2.jpg", basedir + "\\Img1.jpg" };

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

    cout << "xMin: " << xMin << ", xMax: " << xMax << ", yMin: " << yMin << ", yMax: " << yMax << endl;
    cout << width << ", " << height << endl;

    // 5. Initialize the panorama image
    Mat panorama = Mat(height, width, CV_64F);

    // 6. Initialize warped mask images
    vector<Mat> masks_warped;

    // 7. Warp the mask images
    warpMasks(Images, masks_warped, transforms, panorama);

    // 8. Warp the images
    warpImages(Images, masks_warped, transforms, panorama);

    imshow("panorama", panorama);
    waitKey(0);
    destroyWindow("panorama");

    // 9. Initialize the blended panorama images
    Mat pano_feather(panorama.size(), CV_64F);
    Mat pano_multiband(panorama.size(), CV_64F);

    // 10. Blending
    BlendImages(Images, pano_feather, pano_multiband, masks_warped, transforms, panorama.size());

    imshow("feather blended", pano_feather);
    waitKey(0);
    destroyWindow("feather blended");

    imshow("multiband blended", pano_multiband);
    waitKey(0);
    destroyWindow("mutliband blended");

    return 0;
}

// match features in images
void Homography(vector<Mat> Images, vector<Mat> &transforms) {

    // Create a SIFT feature detector object
    Ptr<FeatureDetector> featureDetector = FeatureDetector::create("SIFT");

    // Create a SIFT descriptor extractor object
    Ptr<DescriptorExtractor> descriptorExtractor = DescriptorExtractor::create("SIFT");

    BFMatcher matcher;

    // starting from second image, detect features and match
    for (unsigned int i = 1; i < Images.size(); i++) {
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
        matcher.match(descriptorCurr, descriptorPrev, matchResults);

        // display the matches
        Mat outIm;
        drawMatches(Images[i], keyPointsCurr, Images[i-1], keyPointsPrev, matchResults, outIm, DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
        stringstream ss;
        ss << "Matched Images: " << i-1 << ", " << i;
        imshow(ss.str(), outIm);
        waitKey(0);
        destroyWindow(ss.str());

        vector<Point2d> matchedPointsPrev;
        vector<Point2d> matchedPointsCurr;
        for (unsigned int j = 0; j < matchResults.size(); j++) {
            matchedPointsPrev.push_back(keyPointsPrev[matchResults[j].trainIdx].pt);
            matchedPointsCurr.push_back(keyPointsCurr[matchResults[j].queryIdx].pt);
        }

        // estimate transform between images i and i-1
        Mat T = findHomography(matchedPointsCurr, matchedPointsPrev, CV_RANSAC);

        // compute transform to the ref
        transforms[i] = transforms[i-1] * T;
    }
}

void FindOutputLimits(vector<Mat> Images, vector<Mat> &transforms, int &xMin, int &xMax, int &yMin, int &yMax) {
    // for each image project the corners and find the min/max corner coords
    xMin = numeric_limits<int>::max();
    xMax = numeric_limits<int>::min();
    yMin = numeric_limits<int>::max();
    yMax = numeric_limits<int>::min();

    for (unsigned int i = 0; i < transforms.size(); i++) {
        Mat projected;
        vector<Mat> corners;
        Size s = Images[i].size();

        // top left
        Mat corner = Mat::zeros(3, 1, CV_64F);
        corner.at<double>(2,0) = 1;
        corners.push_back(corner);
        // bottom left
        Mat corner2 = Mat::zeros(3, 1, CV_64F);
        corner2.at<double>(1,0) = s.height - 1;
        corner2.at<double>(2,0) = 1;
        corners.push_back(corner2);
        // top right
        Mat corner4 = Mat::zeros(3, 1, CV_64F);
        corner4.at<double>(0,0) = s.width - 1;
        corner4.at<double>(2,0) = 1;
        corners.push_back(corner4);
        // bottom right
        Mat corner3 = Mat::ones(3, 1, CV_64F);
        corner3.at<double>(0,0) = s.width - 1;
        corner3.at<double>(1,0) = s.height - 1;
        corners.push_back(corner3);

        for (unsigned int j = 0; j < corners.size(); j++) {
            projected = transforms[i] * corners[j];

            // compare for min/max
            if ((int)projected.at<double>(0, 0) < xMin) {
                xMin = (int)projected.at<double>(0, 0);
            }
            if ((int)projected.at<double>(0, 0) > xMax) {
                xMax = (int)projected.at<double>(0, 0);
            }
            if ((int)projected.at<double>(1, 0) < yMin) {
                yMin = (int)projected.at<double>(1, 0);
            }
            if ((int)projected.at<double>(1, 0) > yMax) {
                yMax = (int)projected.at<double>(1, 0);
            }
        }
    }

    // Translate all images so that xMin and yMin become zero
    Mat translation = Mat::eye(3, 3, CV_64F);
    translation.at<double>(0, 2) = -1 * xMin;
    translation.at<double>(1, 2) = -1 * yMin;
    for (unsigned int i = 0; i < transforms.size(); i++) {
        transforms[i] = translation * transforms[i];
    }
}

void warpMasks(vector<Mat> Images, vector<Mat> &masks_warped, vector<Mat> transforms, Mat &panorama) {
    for (unsigned int i = 0; i < Images.size(); i++) {
        // create image masks same size of each image and set values to 255
        Mat mask(Images[i].size(), CV_8U);
        mask.setTo(Scalar(255));

        Mat out(panorama.size(), CV_8U);
        warpPerspective(mask, out, transforms[i], panorama.size());

        imshow("warp mask", out);
        waitKey(0);
        destroyWindow("warp mask");

        masks_warped.push_back(out);
    }
}

void warpImages(vector<Mat> Images, vector<Mat> &masks_warped, vector<Mat> transforms, Mat &panorama) {
    for (unsigned int i = 0; i < Images.size(); i++) {
        // warp image
        Mat out(panorama.size(), Images[i].type());
        warpPerspective(Images[i], out, transforms[i], panorama.size(), 1);

        // copy non-zero pixels using the mask
        out.copyTo(panorama, masks_warped[i]);
    }

//    cout << "displaying panorama" << endl;

//    imshow("panorama", panorama);
//    waitKey(0);
//    destroyWindow("panorama");
}

void BlendImages(vector<Mat> Images, Mat &pano_feather, Mat &pano_multiband, vector<Mat> masks_warped, vector<Mat> transforms, Size pano_size) {
    // Create the feather and multiband blender objects
    detail::FeatherBlender feather;
    detail::MultiBandBlender multiband;

    feather.prepare(Rect(0, 0, pano_feather.cols, pano_feather.rows));
    multiband.prepare(Rect(0, 0, pano_multiband.cols, pano_multiband.rows));

    // feed images to blenders
    for (unsigned int i = 0; i < Images.size(); i++) {
        Mat warped(pano_size, Images[i].type());
        warpPerspective(Images[i], warped, transforms[i], pano_size, 1);
        warped.convertTo(warped, CV_16S);
        feather.feed(warped, masks_warped[i], Point(0, 0));
        multiband.feed(warped, masks_warped[i], Point(0, 0));
    }

    // blend
    Mat empty;
    feather.blend(pano_feather, empty);
    pano_feather.convertTo(pano_feather, CV_8U);
    multiband.blend(pano_multiband, empty);
    pano_multiband.convertTo(pano_multiband, CV_8U);
}
