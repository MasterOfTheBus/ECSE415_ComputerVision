#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/stitching/detail/blenders.hpp>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <limits>

#define REPORT_OUTPUT 0

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

    // Don't Work because of the perspective in the images
//    const string IMG_NAMES[] = { basedir + "\\Img1.jpg", basedir + "\\Img2.jpg", basedir + "\\Img3.jpg", basedir + "\\Img4.jpg", basedir + "\\Img5.jpg", basedir + "\\Img6.jpg" };
//    const string IMG_NAMES[] = {basedir + "\\photo(1).jpg", basedir + "\\photo(2).jpg", basedir + "\\photo(3).jpg", basedir + "\\photo(4).jpg", basedir + "\\photo(5).jpg", basedir + "\\photo(6).jpg"};
//    const string IMG_NAMES[] = {basedir + "\\photo (1).jpg", basedir + "\\photo (2).jpg", basedir + "\\photo (3).jpg", basedir + "\\photo (4).jpg", basedir + "\\photo (5).jpg", basedir + "\\photo (6).jpg"};

    // Work
//        const string IMG_NAMES[] = {basedir + "\\sample1.jpg", basedir + "\\sample2.jpg", basedir + "\\sample3.jpg", basedir + "\\sample4.jpg", basedir + "\\sample5.jpg", basedir + "\\sample6.jpg" };
    const string IMG_NAMES[] = {basedir +  "\\image.jpg", basedir +  "\\image[1].jpg", basedir +  "\\image[2].jpg", basedir +  "\\image[3].jpg", basedir +  "\\image[4].jpg", basedir +  "\\image[5].jpg"};
//    const string IMG_NAMES[] = {basedir +  "\\image.jpg", basedir +  "\\image[5].jpg", basedir +  "\\image[4].jpg", basedir +  "\\image[3].jpg", basedir +  "\\image[2].jpg", basedir +  "\\image[1].jpg"};

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

    if (width > 2000 && height > 700) {
        resize(panorama, panorama, Size(), 0.5, 0.5);
    }

    imshow("panorama", panorama);
    waitKey(0);
    destroyWindow("panorama");

    // 9. Initialize the blended panorama images
    Mat pano_feather(panorama.size(), CV_64F);
    Mat pano_multiband(panorama.size(), CV_64F);

    // 10. Blending
    BlendImages(Images, pano_feather, pano_multiband, masks_warped, transforms, panorama.size());

    if (width > 2000 && height > 700) {
        resize(pano_feather, pano_feather, Size(), 0.5, 0.5);
        resize(pano_multiband, pano_multiband, Size(), 0.5, 0.5);
    }

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
        matcher.match(descriptorPrev, descriptorCurr, matchResults);

        // display the matches
//        Mat outIm;
//        drawMatches(Images[i-1], keyPointsPrev, Images[i], keyPointsCurr, matchResults, outIm, DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
//        stringstream ss;
//        ss << "Matched Images: " << i-1 << ", " << i;
//        imshow(ss.str(), outIm);
//        waitKey(0);
//        destroyWindow(ss.str());

        double min_dist = numeric_limits<double>::max();
        double max_dist = 0;
        for(unsigned int j = 0; j < matchResults.size(); j++ ) {
            double dist = matchResults[j].distance;
            if( dist < min_dist ) min_dist = dist;
            if( dist > max_dist ) max_dist = dist;
        }

        vector<Point2d> matchedPointsPrev;
        vector<Point2d> matchedPointsCurr;
        vector<DMatch> drawMatch;
        for (unsigned int j = 0; j < matchResults.size(); j++) {
            // drop matches that are too far
            if (matchResults[j].distance < min_dist * 2.5) {
                drawMatch.push_back(matchResults[j]);
                matchedPointsPrev.push_back(keyPointsPrev[matchResults[j].queryIdx].pt);
                matchedPointsCurr.push_back(keyPointsCurr[matchResults[j].trainIdx].pt);
            }
        }

        // display the matches
        Mat outIm;
        drawMatches(Images[i-1], keyPointsPrev, Images[i], keyPointsCurr, drawMatch, outIm, DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
        stringstream ss;
        ss << "Matched Images: " << i-1 << ", " << i;
        imshow(ss.str(), outIm);
        waitKey(0);
        destroyWindow(ss.str());

        // estimate transform between images i and i-1
        Mat T = findHomography(matchedPointsCurr, matchedPointsPrev, CV_RANSAC);

        // compute transform to the ref
        transforms[i] = transforms[i-1] * T;

        if (i == Images.size() - 1) {
            cout << "transforms[" << i << "]\n" << transforms[i] << endl;
#if REPORT_OUTPUT

            // calculate the transform using the matched points
            descriptorExtractor->compute(Images[0], keyPointsPrev, descriptorPrev);
            descriptorExtractor->compute(Images[5], keyPointsCurr, descriptorCurr);
            matcher.match(descriptorPrev, descriptorCurr, matchResults);

            min_dist = numeric_limits<double>::max();
            max_dist = 0;
            for(unsigned int j = 0; j < matchResults.size(); j++ ) {
                double dist = matchResults[j].distance;
                if( dist < min_dist ) min_dist = dist;
                if( dist > max_dist ) max_dist = dist;
            }

            matchedPointsPrev.clear();
            matchedPointsCurr.clear();
            drawMatch.clear();
            for (unsigned int j = 0; j < matchResults.size(); j++) {
                // drop matches that are too far
                if (matchResults[j].distance < min_dist * 2.5) {
                    drawMatch.push_back(matchResults[j]);
                    matchedPointsPrev.push_back(keyPointsPrev[matchResults[j].queryIdx].pt);
                    matchedPointsCurr.push_back(keyPointsCurr[matchResults[j].trainIdx].pt);
                }
            }

            // estimate transform between images i and i-1
            T = findHomography(matchedPointsCurr, matchedPointsPrev, CV_RANSAC);

            // compute transform to the ref
            Mat T_disp = transforms[0] * T;

            cout << "T: " << T_disp << endl;
#endif

        }
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
            // normalize the vector so that z coordinate is 1
            double normFactor = 1 / projected.at<double>(2, 0);
            projected = normFactor * projected;

//            cout << "projected" << endl << projected << endl;

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
    cout << "translation:\n" << translation << endl;

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

#if REPORT_OUTPUT
        imshow("warp mask", out);
        waitKey(0);
        destroyWindow("warp mask");
#endif

        masks_warped.push_back(out);
    }
}

void warpImages(vector<Mat> Images, vector<Mat> &masks_warped, vector<Mat> transforms, Mat &panorama) {
    for (unsigned int i = 0; i < Images.size(); i++) {
        // warp image
        Mat out(panorama.size(), Images[i].type());
        warpPerspective(Images[i], out, transforms[i], panorama.size(), 1);

#if REPORT_OUTPUT
        imshow("warp image", out);
        waitKey(0);
        destroyWindow("warp image");
#endif

        // copy non-zero pixels using the mask
        out.copyTo(panorama, masks_warped[i]);
    }
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
        warpPerspective(Images[i], warped, transforms[i], pano_size, 1, BORDER_REPLICATE);
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
