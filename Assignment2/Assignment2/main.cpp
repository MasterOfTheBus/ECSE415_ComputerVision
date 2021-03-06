#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <limits>
#include <stdlib.h>
#include <time.h>

#define NO_DISCARD 1
#define TRAINING 0

using namespace cv;
using namespace std;

/* Helper class declaration and definition */
class Caltech101
{
public:
    Caltech101(string datasetPath, const int numTrainingImages, const int numTestImages)
	{	
		cout << "Loading Caltech 101 dataset" << endl;
		numImagesPerCategory = numTrainingImages + numTestImages;

		// load "Categories.txt"
		ifstream infile(datasetPath + "/" + "Categories.txt");
		cout << "\tChecking Categories.txt" << endl;
		if (!infile.is_open())
		{
			cout << "\t\tError: Cannot find Categories.txt in " << datasetPath << endl;
			return;
		}
		cout << "\t\tOK!" << endl;

		// Parse category names
		cout << "\tParsing category names" << endl;
		string catname;
		while (getline(infile, catname))
		{
			categoryNames.push_back(catname);
		}
		cout << "\t\tdone!" << endl;

		// set num categories
		int numCategories = (int)categoryNames.size();

		// initialize outputs size
		trainingImages = vector<vector<Mat>>(numCategories);
		trainingAnnotations = vector<vector<Rect>>(numCategories);
		testImages = vector<vector<Mat>>(numCategories);
		testAnnotations = vector<vector<Rect>>(numCategories);

		// generate training and testing indices
		randomShuffle();		

		// Load data
		cout << "\tLoading images and annotation files" << endl;
		string imgDir = datasetPath + "/" + "Images";
		string annotationDir = datasetPath + "/" + "Annotations";
		for (int catIdx = 0; catIdx < categoryNames.size(); catIdx++)
		//for (int catIdx = 0; catIdx < 1; catIdx++)
		{
			string imgCatDir = imgDir + "/" + categoryNames[catIdx];
			string annotationCatDir = annotationDir + "/" + categoryNames[catIdx];
			for (int fileIdx = 0; fileIdx < numImagesPerCategory; fileIdx++)
			{
				// use shuffled training and testing indices
				int shuffledFileIdx = indices[fileIdx];
				// generate file names
				stringstream imgFilename, annotationFilename;
				imgFilename << "image_" << setfill('0') << setw(4) << shuffledFileIdx << ".jpg";
				annotationFilename << "annotation_" << setfill('0') << setw(4) << shuffledFileIdx << ".txt";

				// Load image
				string imgAddress = imgCatDir + '/' + imgFilename.str();
				Mat img = imread(imgAddress, CV_LOAD_IMAGE_COLOR);
				// check image data
				if (!img.data)
				{
					cout << "\t\tError loading image in " << imgAddress << endl;
					return;
				}

				// Load annotation
				string annotationAddress = annotationCatDir + '/' + annotationFilename.str();
				ifstream annotationIFstream(annotationAddress);
				// Checking annotation file
				if (!annotationIFstream.is_open())
				{
					cout << "\t\tError: Error loading annotation in " << annotationAddress << endl;
					return;
				}
				int tl_col, tl_row, width, height;
				Rect annotRect;
				while (annotationIFstream >> tl_col >> tl_row >> width >> height)
				{
					annotRect = Rect(tl_col - 1, tl_row - 1, width, height);					
				}

				// Split training and testing data
				if (fileIdx < numTrainingImages)
				{
					// Training data
					trainingImages[catIdx].push_back(img);
					trainingAnnotations[catIdx].push_back(annotRect);
				}
				else
				{
					// Testing data
					testImages[catIdx].push_back(img);
					testAnnotations[catIdx].push_back(annotRect);
				}				
			}			
		}
		cout << "\t\tdone!" << endl;		
		successfullyLoaded = true;
		cout << "Dataset successfully loaded: " << numCategories << " categories, " << numImagesPerCategory  << " images per category" << endl << endl;
	}

	bool isSuccessfullyLoaded()	{  return successfullyLoaded; }

	void dispTrainingImage(int categoryIdx, int imageIdx)
	{		
		Mat image = trainingImages[categoryIdx][imageIdx];
		Rect annotation = trainingAnnotations[categoryIdx][imageIdx];
		rectangle(image, annotation, Scalar(255, 0, 255), 2);
		imshow("Annotated training image", image);
		waitKey(0);
		destroyWindow("Annotated training image");
	}
	
	void dispTestImage(int categoryIdx, int imageIdx)
	{
		Mat image = testImages[categoryIdx][imageIdx];
		Rect annotation = testAnnotations[categoryIdx][imageIdx];
		rectangle(image, annotation, Scalar(255, 0, 255), 2);
		imshow("Annotated test image", image);
		waitKey(0);
		destroyWindow("Annotated test image");
	}

	vector<string> categoryNames; 
	vector<vector<Mat>> trainingImages;
	vector<vector<Rect>> trainingAnnotations;
	vector<vector<Mat>> testImages;
	vector<vector<Rect>> testAnnotations;

private:
	bool successfullyLoaded = false;
	int numImagesPerCategory;
	vector<int> indices;
	void randomShuffle()
	{
		// set init values
		for (int i = 1; i <= numImagesPerCategory; i++) indices.push_back(i);

		// permute using built-in random generator
		random_shuffle(indices.begin(), indices.end());		
	}
};

/* Function prototypes */
void Train(const Caltech101 &Dataset, Mat &codeBook, vector<vector<Mat>> &imageDescriptors, const int numCodewords);
void Test(const Caltech101 &Dataset, const Mat codeBook, const vector<vector<Mat>> imageDescriptors);

int main(void)
{
	/* Initialize OpenCV nonfree module */
	initModule_nonfree();

	/* Put the full path of the Caltech 101 folder here */
    const string datasetPath = "C:\\Users\\Lazy\\Documents\\GitHub\\Computer_Vision\\Assignment2\\Caltech 101\\Caltech 101";

	/* Set the number of training and testing images per category */
	const int numTrainingData = 40;
	const int numTestingData = 2;

	/* Set the number of codewords*/
    const int numCodewords = 100;

	/* Load the dataset by instantiating the helper class */
	Caltech101 Dataset(datasetPath, numTrainingData, numTestingData);

	/* Terminate if dataset is not successfull loaded */
	if (!Dataset.isSuccessfullyLoaded())
	{
		cout << "An error occurred, press Enter to exit" << endl;
		getchar();
        return -1;
	}	
	
	/* Variable definition */
	Mat codeBook;	
    vector<vector<Mat>> imageDescriptors;

    ofstream file;
//    stringstream ss;
//    ss << "C:\\Users\\Lazy\\Documents\\GitHub\\Computer_Vision\\Assignment2\\to_matlab" << i << ".txt" << endl;
    file.open("test_file.txt");
    file << "HI\n";
    file.close();

	/* Training */	
//    Train(Dataset, codeBook, imageDescriptors, numCodewords);

    cout << "Trained" << endl;

	/* Testing */	
//    Test(Dataset, codeBook, imageDescriptors);

    cout << "All Done" << endl;

    return 0;
}

/* Train BoW */
void Train(const Caltech101 &Dataset, Mat &codeBook, vector<vector<Mat>> &imageDescriptors, const int numCodewords)
{
    // Create a SIFT feature detector object
    Ptr<FeatureDetector> featureDetector = FeatureDetector::create("SIFT");

    // Create a SIFT descriptor extractor object
    Ptr<DescriptorExtractor> descriptorExtractor = DescriptorExtractor::create("SIFT");

    // Mat object to store all the SIFT descriptors of all training categories
    Mat D;

    // loop for each category
    for (unsigned int i = 0; i < Dataset.trainingImages.size(); i++) {
        // each image of each category
        for (unsigned int j = 0; j < Dataset.trainingImages[i].size(); j++) {
            Mat I = Dataset.trainingImages[i][j];

            // detect key points
            vector<KeyPoint> keyPoints;
            featureDetector->detect(I, keyPoints);

#if NO_DISCARD
            // get the annotation rectangle
            Rect annotation = Dataset.trainingAnnotations[i][j];

            int bx = annotation.width + annotation.tl().x;
//            int by = annotation.tl().y;
//            int dx = annotation.tl().x;
            int dy = annotation.height + annotation.tl().x;

            int bax = annotation.width;
//            int bay = 0;
//            int dax = 0;
            int day = annotation.height;

            // discard keypoints ouside the annotation rectangle
            for (unsigned int k = 0; k < keyPoints.size(); k++) {
                // outside of rectangle, discard
                if (!annotation.contains(keyPoints[k].pt)) {
                    keyPoints.erase(keyPoints.begin()+k);
                }
            }
#endif

//            if (j == 15) {
//                // test to see the key points on the image
//                Mat outI;
//                drawKeypoints(I, keyPoints, outI);
//                CvScalar color = {0.0, 0.0, 255.0, 0.0};
//                rectangle(outI, annotation, color);
//                imshow("Annotated test with key points", outI);
//                waitKey(0);
//            }

            // compute SIFT descriptors
            Mat descriptor;
            descriptorExtractor->compute(I, keyPoints, descriptor);

            // Add descriptors to D
            D.push_back(descriptor);
        }
    }

    cout << "computed descriptors, creating the codebook" << endl;

    // create a codebook
    BOWKMeansTrainer trainer = BOWKMeansTrainer(numCodewords);
    trainer.add(D);
    codeBook = trainer.cluster();

    cout << "clustered, generating histograms" << endl;

    // Represent the object categories using the codebook
    // BOW histogram for each image
    Ptr<DescriptorMatcher> descriptorMatcher = DescriptorMatcher::create("BruteForce");
    Ptr<BOWImgDescriptorExtractor> bowDExtractor = new BOWImgDescriptorExtractor(descriptorExtractor, descriptorMatcher);
    bowDExtractor->setVocabulary(codeBook);

    // loop for each category
    for (unsigned int i = 0; i < Dataset.trainingImages.size(); i++) {
        // each image of each category
        vector<Mat> category_descriptors;
        Mat average_mat = Mat::zeros(1, numCodewords, CV_32F);
        for (unsigned int j = 0; j < Dataset.trainingImages[i].size(); j++) {
            Mat I = Dataset.trainingImages[i][j];

            // detect key points
            vector<KeyPoint> keyPoints;
            featureDetector->detect(I, keyPoints);

            // discard keypoints ouside the annotation rectangle
#if NO_DISCARD
            Rect annotation = Dataset.trainingAnnotations[i][j];
            for (unsigned int k = 0; k < keyPoints.size(); k++) {
                // outside of rectangle, discard
                if (!annotation.contains(keyPoints[k].pt)) {
                    keyPoints.erase(keyPoints.begin()+k);
                }
            }
#endif

            // Compute histogram representation and store in descriptors
            //Mat histogram;
            Mat histogram;
            bowDExtractor->compute2(I, keyPoints, histogram);
            category_descriptors.push_back(histogram);

            average_mat += histogram;
        }
        imageDescriptors.push_back(category_descriptors);

        average_mat = average_mat / Dataset.trainingImages[i].size();

#if 0
        ofstream file;
        stringstream ss;
        ss << "C:\\Users\\Lazy\\Documents\\GitHub\\Computer_Vision\\Assignment2\\to_matlab" << i << ".txt" << endl;
        file.open(ss.str());
        for (int m = 0; m < average_mat.rows; m++) {
            for (int n = 0; n < average_mat.cols; n++) {
                file << average_mat.at<double>(m, n) << " ";
            }
            file << "\n";
        }
        file.close();
#endif
    }
}

/* Test BoW */
void Test(const Caltech101 &Dataset, const Mat codeBook, const vector<vector<Mat>> imageDescriptors)
{
    srand(time(NULL));

    // Create a SIFT feature detector object
    Ptr<FeatureDetector> featureDetector = FeatureDetector::create("SIFT");

    // Create a SIFT descriptor extractor object
    Ptr<DescriptorExtractor> descriptorExtractor = DescriptorExtractor::create("SIFT");

    // BOW histogram for each image
    Ptr<DescriptorMatcher> descriptorMatcher = DescriptorMatcher::create("BruteForce");
    Ptr<BOWImgDescriptorExtractor> bowDExtractor = new BOWImgDescriptorExtractor(descriptorExtractor, descriptorMatcher);

    bowDExtractor->setVocabulary(codeBook);

//    vector< vector<unsigned int> > compared_categories;
    int numCorrect = 0;
//    int printcount = 0;

    // loop for each category
    for (unsigned int i = 0; i < Dataset.testImages.size(); i++) {
        // each image of each category
        cout << i << endl;
        for (unsigned int j = 0; j < Dataset.testImages[i].size(); j++) {
#if TRAINING
            Mat I = Dataset.trainingImages[i][j];
#else
            Mat I = Dataset.testImages[i][j];
#endif

            // detect key points
            vector<KeyPoint> keyPoints;
            featureDetector->detect(I, keyPoints);

#if NO_DISCARD
            // discard keypoints ouside the annotation rectangle
            Rect annotation = Dataset.testAnnotations[i][j];

//            if (printcount < 2 && (rand() % 5 + 1) == 3) {
//                printcount++;
//                // test to see the key points on the image
//                Mat outI;
//                drawKeypoints(I, keyPoints, outI);
//                CvScalar color = {1.0, 0.0, 0.0, 0.0};
//                rectangle(outI, annotation, color);
//                stringstream ss;
//                ss << "Category " << i << " Image " << j;
//                imshow(ss.str(), outI);
//                waitKey(0);
//                destroyWindow(ss.str());
//            }

            for (unsigned int k = 0; k < keyPoints.size(); k++) {
                // outside of rectangle, discard
                if (!annotation.contains(keyPoints[k].pt)) {
                    keyPoints.erase(keyPoints.begin()+k);
                }
            }
#endif

            // Compute histogram representation
            Mat histogram;
            bowDExtractor->compute2(I, keyPoints, histogram);

            // compare and find the best matching histogram
            double best_dist = numeric_limits<double>::max();
            unsigned int best_m = 0;
//            unsigned int best_n = 0;
            for (unsigned int m = 0; m < imageDescriptors.size(); m++) {
                for (unsigned int n = 0; n < imageDescriptors[m].size(); n++) {
                    double dist = norm(histogram, imageDescriptors[m][n]);
                    if (dist < best_dist) {
                        best_dist = dist;
                        best_m = m;
//                        best_n = n;
                    }
                }
            }

            // assign the category index
            // compared_categories[i][j] = best_m;
            cout << "category: " << i << ": " << Dataset.categoryNames[i] << " image " << j << " marked as "
                 << Dataset.categoryNames[best_m] << endl;
            if (best_m == i) numCorrect++;
        }
    }

    double ratio = double(numCorrect) / double(Dataset.testImages.size() * Dataset.testImages[0].size());
    cout << ratio << endl;
}
