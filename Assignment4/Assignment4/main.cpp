#include <opencv2/opencv.hpp>
#include <string>
#include <iostream>
#include <vector>
#include "stdlib.h"

#define DEBUGGING_KMEANS 0
#define INTENSITY 0

using namespace std;
using namespace cv;

Mat calc_kmeans(Mat img, int rect[]);
Mat calc_gmm(Mat img, int rect[]);
Mat calc_graphCut(Mat img, int rect[]);
Mat maskImg(Mat img, Mat mask);

int main(/*int argc, char *argv[]*/)
{
    const string base_dir = "C:\\Users\\Lazy\\Documents\\GitHub\\Computer_Vision\\Assignments\\Assignment4\\";
    string image_names[] = {base_dir + "100_0109.png",
                            base_dir + "b4nature_animals_land009.png",
                            base_dir + "cheeky_penguin.png"};

    string gt_names[] = {base_dir + "100_0109_groundtruth.png",
                         base_dir + "b4nature_animals_land009_groundtruth.png",
                         base_dir + "cheeky_penguin_groundtruth.png"};

    string rect_file = base_dir + "rect.txt";

    // load the input image
    Mat img = imread(image_names[0], CV_LOAD_IMAGE_COLOR);

    // load the ground truth image
    Mat gtmask = imread(gt_names[0], CV_LOAD_IMAGE_COLOR);

    // load the rect values from file
    FILE* file = fopen(rect_file.c_str(), "r");

    if (file != NULL) {
        int rect_points[4];
        char* line = (char*)malloc(100);
        if (fgets(line, 100, file) != NULL) {
            rect_points[0] = atoi(strtok(line, " \t"));
            for (int i = 1; i < 4; i++) {
                rect_points[i] = atoi(strtok(NULL, " \t"));
            }

            Mat kmeans_mask = calc_kmeans(img, rect_points);
            Mat out_img = maskImg(img, kmeans_mask);
            // display the results
            imshow("kmeans", out_img);
            waitKey(0);
            destroyWindow("kmeans");

//            Mat gmm_mask = calc_gmm(img, rect_points);
//            out_img = maskImg(img, gmm_mask);
//            // display the results
//            imshow("gmm", out_img);
//            waitKey(0);
//            destroyWindow("gmm");

            Mat graphCut_mask = calc_graphCut(img, rect_points);
            out_img = maskImg(img, graphCut_mask);
            // display the results
            imshow("graph cut", out_img);
            waitKey(0);
            destroyWindow("graph cut");
        }

    }

    return 0;
}

Mat calc_kmeans(Mat img, int rect[]) {
    int H = img.size().height;
    int W = img.size().width;

    // convert the img to double before we use it
    img.convertTo(img, CV_32F);

    // create the corresponding user drawn rect
    Rect_<double> user_rect(rect[0], rect[1], rect[2], rect[3]);

#if DEBUGGING_KMEANS
    CvScalar color = {0.0, 0.0, 255.0, 0.0};
    Mat outIm = img;
    rectangle(outIm, user_rect, color);
    imshow("img with rect", outIm);
    waitKey(0);
    destroyWindow("img with rect");
#endif

    // feature matrix
//    Mat feature_mat = img.reshape(1, H*W);
#if INTENSITY
    cvtColor(img, img, CV_RGB2GRAY);
#endif
    Mat feature_mat = Mat::zeros(H*W, 3, CV_32F);
    for (int col = 0; col < W; col++) {
        for (int row = 0; row < H; row++) {
            feature_mat.at<float>(col*H+row,0) = img.at<Vec3f>(row, col)[2]; // red
            feature_mat.at<float>(col*H+row,1) = img.at<Vec3f>(row, col)[1]; // green
            feature_mat.at<float>(col*H+row,2) = img.at<Vec3f>(row, col)[0]; // blue
        }
    }

    // initial label matrix that has points assigned for inside and outside
    Mat init_label = Mat::zeros(H*W, 1, CV_32S);
    for (int x = (int)user_rect.x; x < (int)(user_rect.x + user_rect.width); x++) {
        for (int y = (int)user_rect.y; y < (int)(user_rect.y + user_rect.height); y++) {
            init_label.at<int>(x*H+y, 0) = 1;
        }
    }

#if DEBUGGING_KMEANS
    Mat label_out = Mat::zeros(H, W, CV_8U);
    for (int i = 0; i < W; i++) {
        for (int j = 0; j < H; j++) {
            label_out.at<char>(j, i) = 255*init_label.at<int>(i*H+j, 0);
        }
    }
    imshow("label", label_out);
    waitKey(0);
    destroyWindow("label");
#endif

    // run kmeans
    int k = 2, attempts = 3;
    kmeans(feature_mat, k, init_label,
           TermCriteria( CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 1000, 0.0001),
           attempts, KMEANS_USE_INITIAL_LABELS);

#if DEBUGGING_KMEANS
    label_out = Mat::zeros(H, W, CV_8U);
    for (int i = 0; i < W; i++) {
        for (int j = 0; j < H; j++) {
            label_out.at<char>(j, i) = 255*init_label.at<int>(i*H+j, 0);
        }
    }
    imshow("label after convergence", label_out);
    waitKey(0);
    destroyWindow("label after convergence");
#endif

    // create the foreground mask matrix
    // TODO: use reshape here and above
    Mat foreground_mask = Mat::zeros(H, W, CV_8UC1);
    for (int i = 0; i < W; i++) {
        for (int j = 0; j < H; j++) {
            foreground_mask.at<char>(j, i) = init_label.at<int>(i*H+j,0);
        }
    }

    return foreground_mask;
}

Mat calc_gmm(Mat img, int rect[]) {
    int H = img.size().height;
    int W = img.size().width;

    // create the corresponding user drawn rect
    Rect_<double> user_rect(rect[0], rect[1], rect[2], rect[3]);

#if INTENSITY
    cvtColor(img, img, CV_RGB2Lab);
#endif

    // Create Expectation Maximization object, feature matrix, initial means matrix
    EM em(2);
    Mat feature_mat = img.reshape(1, H*W);
    feature_mat.convertTo(feature_mat, CV_64F);
    Mat initial_means = Mat::zeros(2, 3, CV_64F);

    // Initial covariation matrices
    vector<Mat> covar;
    covar.push_back(Mat::zeros(2,2,CV_64F));
    covar.push_back(Mat::zeros(2,2,CV_64F));

    Mat inbox, outbox;
    outbox.push_back((img.rowRange(0, user_rect.y)).reshape(0, W*user_rect.y));
    outbox.push_back((img.rowRange(user_rect.y+user_rect.height, H)).reshape(0, W*(H-user_rect.height-user_rect.y)));

//    for (int row = 0; row < H; row++) {
//        for (int col = 0; col < W; col++) {
//            if () {

//            }
//        }
//    }

    calcCovarMatrix(inbox, covar[0], initial_means.row(0), CV_COVAR_NORMAL | CV_COVAR_ROWS);
    calcCovarMatrix(outbox, covar[1], initial_means.row(1), CV_COVAR_NORMAL | CV_COVAR_ROWS);

    // Initial weight
    Mat init_weight = Mat::zeros(1, 2, CV_32F);
    init_weight.at<float>(0,0) = user_rect.area() / (H*W);
    init_weight.at<float>(0,1) = 1 - init_weight.at<float>(0,0);
    Mat labels;

    em.trainE(feature_mat, initial_means, covar, init_weight, noArray(), labels);

    return labels;
}

Mat calc_graphCut(Mat img, int rect[]) {
    // create the corresponding user drawn rect
    Rect_<double> user_rect(rect[0], rect[1], rect[2], rect[3]);

    Mat mask, bgdModel, fgdModel;

#if INTENSITY
    cvtColor(img, img, CV_RGB2GRAY);
#endif

    grabCut(img, mask, user_rect, bgdModel, fgdModel, 3, GC_INIT_WITH_RECT);

    Mat foreground_mask = mask & GC_FGD;

    return foreground_mask;
}

Mat maskImg(Mat img, Mat mask) {  
    int H = img.size().height;
    int W = img.size().width;
    Mat out_img = Mat::zeros(H, W, CV_8UC3);
    for (int i = 0; i < img.channels(); i++) {
        Mat channelMat = Mat::zeros(H, W, CV_8UC1);
        // img[i] -> channelMat[0]
        int fromTo[] = {i, 0};
        mixChannels(&img, 1, &channelMat, 1, fromTo, 1);
        channelMat = channelMat.mul(mask);
        // channelMat[0] -> out_img[i]
        int fromTo2[] = {0, i};
        mixChannels(&channelMat, 1, &out_img, 1, fromTo2, 1);
    }
    return out_img;
}
