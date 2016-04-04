#include <opencv2/opencv.hpp>
#include <string>
#include <iostream>
#include <vector>
#include "stdlib.h"

#define INTENSITY 0

using namespace std;
using namespace cv;

Mat calc_kmeans(Mat img, int rect[]);
Mat calc_gmm(Mat img, int rect[]);
Mat calc_graphCut(Mat img, int rect[]);
Mat maskImg(Mat img, Mat mask);

int main()
{
    const string base_dir = "C:\\Users\\Lazy\\Documents\\GitHub\\Computer_Vision\\Assignments\\Assignment4\\";
    string image_names[] = {base_dir + "100_0109.png",
                            base_dir + "b4nature_animals_land009.png",
                            base_dir + "cheeky_penguin.png"};

    string gt_names[] = {base_dir + "100_0109_groundtruth.png",
                         base_dir + "b4nature_animals_land009_groundtruth.png",
                         base_dir + "cheeky_penguin_groundtruth.png"};

    string rect_files[] = {base_dir + "100_0109_rect.txt",
                           base_dir + "b4nature_animals_land009_rect.txt",
                           base_dir + "cheeky_penguin_rect.txt"};

    for (int index = 0; index < 3; index++) {
        // load the input image
        Mat img = imread(image_names[index], CV_LOAD_IMAGE_COLOR);

        // load the ground truth image
        //    Mat gtmask = imread(gt_names[0], CV_LOAD_IMAGE_COLOR);

        // load the rect values from file
        FILE* file = fopen(rect_files[index].c_str(), "r");

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

                Mat gmm_mask = calc_gmm(img, rect_points);
                out_img = maskImg(img, gmm_mask);
                // display the results
                imshow("gmm", out_img);
                waitKey(0);
                destroyWindow("gmm");

                Mat graphCut_mask = calc_graphCut(img, rect_points);
                out_img = maskImg(img, graphCut_mask);
                // display the results
                imshow("graph cut", out_img);
                waitKey(0);
                destroyWindow("graph cut");
            }

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

    // feature matrix
#if INTENSITY
    cvtColor(img, img, CV_RGB2GRAY);
#endif
    Mat feature_mat = img.reshape(1, H*W);

    // initial label matrix that has points assigned for inside and outside
    Mat init_label = Mat::zeros(H*W, 1, CV_32S);
    for (int x = (int)user_rect.x; x < (int)(user_rect.x + user_rect.width); x++) {
        for (int y = (int)user_rect.y; y < (int)(user_rect.y + user_rect.height); y++) {
            init_label.at<int>(x*H+y, 0) = 1;
        }
    }

    // run kmeans
    int k = 2, attempts = 3;
    kmeans(feature_mat, k, init_label,
           TermCriteria( CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 1000, 0.0001),
           attempts, KMEANS_USE_INITIAL_LABELS);

    Mat label = init_label.reshape(0, H);
    label.convertTo(label, CV_8UC1);

    // why are the labels flipped?
    char flipper = 1;
    label = label ^ flipper;

    return label;
}

Mat calc_gmm(Mat img, int rect[]) {
    int H = img.size().height;
    int W = img.size().width;
    int clusters = 2, dim = 3;

    // create the corresponding user drawn rect
    Rect_<double> user_rect(rect[0], rect[1], rect[2], rect[3]);

#if INTENSITY
    cvtColor(img, img, CV_RGB2GRAY);
#endif

    img.convertTo(img, CV_64F);

    // Create Expectation Maximization object, feature matrix, initial means matrix
    EM em(clusters, EM::COV_MAT_DIAGONAL, TermCriteria(CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 1000, 0.0001));
    Mat feature_mat = img.reshape(1, H*W);
    Mat initial_means = Mat::zeros(clusters, dim, CV_64F);

    // Initial covariation matrices
    vector<Mat> covar;
    covar.push_back(Mat::zeros(dim,dim,CV_64F));
    covar.push_back(Mat::zeros(dim,dim,CV_64F));

    // push the background pixels to a matrix
    Mat outbox;
    outbox.push_back(img.rowRange(0, user_rect.y));
    outbox = outbox.reshape(0, user_rect.y * W);

    // the pixels that are on the side of the foreground pixels
    for (int row = user_rect.y; row <= user_rect.y+user_rect.height; row++) {
        for (int col = 0; col < W; col++) {
            if ((col < user_rect.x || col >= user_rect.x+user_rect.width)) {
                outbox.push_back(img.at<Vec3d>(row, col));
            }
        }
    }
    Mat last_rows = img.rowRange(user_rect.y+user_rect.height, H);
    outbox.push_back(last_rows.reshape(0, (H-user_rect.y-user_rect.height) * W));
    outbox = outbox.reshape(1, 0);

    Mat inbox_submat = img(user_rect);
    Mat inbox = inbox_submat.clone(); // make sure that the data is continuous before reshaping
    inbox = inbox.reshape(1, inbox.rows * inbox.cols); // reshape so that we can pass to calcCovarMatrix

    calcCovarMatrix(inbox, covar[0], initial_means.row(0), CV_COVAR_NORMAL | CV_COVAR_ROWS);
    calcCovarMatrix(outbox, covar[1], initial_means.row(1), CV_COVAR_NORMAL | CV_COVAR_ROWS);

    // why does this work? what does it do?
    covar[0] /= (H*W);
    covar[1] /= (H*W);

    // Initial weight
    Mat init_weight = Mat::zeros(1, clusters, CV_64F);
    init_weight.at<double>(0,0) = user_rect.area() / (double)(H*W);
    init_weight.at<double>(0,1) = 1 - init_weight.at<double>(0,0);
    Mat labels;

    em.trainE(feature_mat, initial_means, covar, init_weight, noArray(), labels);

    // reshape the labels so that it's an H * W matrix
    labels = labels.reshape(0, H);
    labels.convertTo(labels, CV_8UC1);

    // for some reason the labelling got flipped around?
    char flipper = 1;
    labels = labels ^ flipper;

    return labels;
}

Mat calc_graphCut(Mat img, int rect[]) {
    // create the corresponding user drawn rect
    Rect_<double> user_rect(rect[0], rect[1], rect[2], rect[3]);

    Mat mask, bgdModel, fgdModel;

#if INTENSITY
    cvtColor(img, img, CV_RGB2Lab);
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
