#include <opencv2/opencv.hpp>
#include <string>
#include <iostream>
#include "stdlib.h"

using namespace std;
using namespace cv;

Mat calc_kmeans(Mat img, int rect[]);

int main(int argc, char *argv[])
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
    int H = img.size().height;
    int W = img.size().width;

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

            imshow("kmeans mask", kmeans_mask);
            waitKey(0);
            destroyWindow("kmeans mask");

            // display the results
            Mat out_img = Mat::zeros(H, W, CV_8UC3);
            for (int i = 0; i < img.channels(); i++) {
                Mat channelMat = Mat::zeros(H, W, CV_8UC1);
                // img[i] -> channelMat[0]
                int fromTo[] = {i, 0};
                mixChannels(&img, 1, &channelMat, 1, fromTo, 1);
                channelMat = channelMat.mul(kmeans_mask);
                // channelMat[0] -> out_img[i]
                int fromTo2[] = {0, i};
                mixChannels(&channelMat, 1, &out_img, 1, fromTo2, 1);
            }
            imshow("kmeans", out_img);
            waitKey(0);
            destroyWindow("kmeans");
        }

    }

    return 0;
}

Mat calc_kmeans(Mat img, int rect[]) {
    int H = img.size().height;
    int W = img.size().width;

    // create the corresponding user drawn rect
    Rect_<double> user_rect(rect[0], rect[1], rect[2], rect[3]);

//    CvScalar color = {0.0, 0.0, 255.0, 0.0};
//    rectangle(img, user_rect, color);
//    imshow("img with rect", img);
//    waitKey(0);
//    destroyWindow("img with rect");

    // feature matrix
    Mat feature_mat = Mat::zeros(H*W, 3, CV_32F);
    for (int col = 0; col < W; col++) {
        for (int row = 0; row < H; row++) {
            feature_mat.at<double>(col*H+row,0) = img.at<Vec3d>(row, col)[2]; // red
            feature_mat.at<double>(col*H+row,1) = img.at<Vec3d>(row, col)[1]; // green
            feature_mat.at<double>(col*H+row,2) = img.at<Vec3d>(row, col)[0]; // blue
        }
    }

    // initial label matrix that has points assigned for inside and outside
    Mat init_label = Mat::zeros(H*W, 1, CV_16U);
    for (int x = (int)user_rect.x; x < (int)(user_rect.x + user_rect.width); x++) {
        for (int y = (int)user_rect.y; y < (int)(user_rect.y + user_rect.height); y++) {
            init_label.at<int>(x*H+y, 0) = 1;
        }
    }

    // run kmeans
    int k = 2, attempts = 3;
//    TermCriteria criteria;
//    criteria.type = TermCriteria::EPS;
//    criteria.epsilon = 0.01;
    kmeans(feature_mat, k, init_label,
           TermCriteria( CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 10, 1.0),
           attempts, KMEANS_PP_CENTERS);

    // create the foreground mask matrix
    Mat foreground_mask = Mat::zeros(H, W, CV_8UC1);
    for (int i = 0; i < H; i++) {
        for (int j = 0; j < W; j++) {
            foreground_mask.at<uchar>(i, j) = init_label.at<unsigned int>(i*H+j,0);
        }
    }

    return foreground_mask;
}
