#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/stitching/detail/blenders.hpp>

// Functions prototypes
void Homography(…);
void FindOutputLimits(…);
void warpMasks(…);
void warpImages(…);
void BlendImages(…);

int main()
{
    // Initialize OpenCV nonfree module
    initModule_nonfree();
    // Set the dir/name of each image
    const int NUM_IMAGES = 2;
    const string IMG_NAMES[] = { "Img1.jpg", " Img2.jpg", " Img3.jpg" };

    // Load the images
    vector<Mat> Images;
    for (int i = 0; i < NUM_IMAGES; i++) {
        Images.push_back(imread(IMG_NAMES[i]));
    }

    // 1. Initialize all the transforms to the identity matrix

    // 2. Calculate the transformation matrices
    Homography(Images, transforms);

    // 3. Compute the min and max limits of the transformations
    int xMin, xMax, yMin, yMax;
    FindOutputLimits(Images, transforms, xMin, xMax, yMin, yMax);

    // 4. Compute the size of the panorama

    // 5. Initialize the panorama image

    // 6. Initialize warped mask images

    // 7. Warp the mask images
    warpMasks(Images, masks_warped, transforms, panorama);

    // 8. Warp the images
    warpImages(Images, masks_warped, transforms, panorama);

    // 9. Initialize the blended panorama images

    // 10. Blending
    BlendImages(Images, pano_feather, pano_multiband, masks_warped, transforms);

    return 0;
}

void Homography(…) {

}

void FindOutputLimits(…) {

}

void warpMasks(…) {

}

void warpImages(…) {

}

void BlendImages(…) {

}
