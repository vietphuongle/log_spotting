/*
 * main.cpp
 *
 *  Created on: Jul 17, 2023
 *      Author: levietphuong
 */




#include <iostream>
#include <fstream>
#include <string>
#include "opencv2/core.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;

struct Element{
	Mat image;
	//string name;
	vector<KeyPoint> keypoints;
	Mat descriptors;
	//int label;
};

vector<Element> logs;

int nb_logs, nb_images;
const string path("");
const string path_logs(path+"logs");
const string path_images(path+"images");
//const string path_rs(path+"rs");
const string list_log_names(path+"list_logs.txt");

Element keypoint_extraction(string name){
	Element e;
	Mat img = imread(name, IMREAD_GRAYSCALE);
	resize(img, e.image, Size(), 0.5, 0.5, INTER_LINEAR);
	Ptr<SIFT> detector = SIFT::create();
	vector<KeyPoint> keypoints;
	Mat descriptors;
	detector->detectAndCompute(e.image, noArray(), e.keypoints, e.descriptors);
	//e.image = img;
	//e.keypoints=keypoints;
	//e.descriptors=descriptors;
	//e.name=name;
	//e.label=0;
	return e;
}

void read_logs(){
	//read logs
	ifstream ifile(list_log_names);
	ifile >> nb_logs;
	for(int i=0;i<nb_logs;i++){
		string log_name;
		ifile >> log_name;
		//cout << log_name << endl;
		logs.push_back(keypoint_extraction(path_logs+"/"+log_name));
		//string sub1 = log_name.substr(2);
		//logs[count].label = stoi(sub1.substr(0,sub1.find_first_of("_")));
		//cout << logs[count].name << "-" << logs[count].keypoints.size() << "-" << logs[count].label<< endl;
	}
	ifile.close();
}

Element read_image(string image_name){
	Element i;
	i = keypoint_extraction(image_name);
	//i.label = stoi(image_name.substr(0,image_name.find_first_of("_")));
	//cout << i.name << "-" << i.keypoints.size() << "-" << i.label<< endl;
	return i;
}

bool isNearlyRectangle(const vector<Point2f>& corners, double angleThreshold) {
    // checking 4 corners is a nearly rectangle
	// 4 angels are 90 degree +/- angleThreshold
	// Ensure we have exactly four corners
    if (corners.size() != 4) {
        return false;
    }
    Point point1=corners[0];
    Point point2=corners[1];
    Point point3=corners[2];
    Point point4=corners[3];

    // Calculate 1st angle
    Point edge1 = point2 - point1;
    Point edge2 = point3 - point2;

    double dotProduct = edge1.dot(edge2);
    double edge1Magnitude = norm(edge1);
    double edge2Magnitude = norm(edge2);
    double angle1 = acos(dotProduct / (edge1Magnitude * edge2Magnitude)) * 180.0 / CV_PI;
    if (abs(angle1 - 90.0) > angleThreshold) {
    	//cout << "angle1: " << angle1 << endl;
    	return false;
    }

    // Calculate 2nd angle
    Point edge3 = point4 - point3;

    dotProduct = edge2.dot(edge3);
    double edge3Magnitude = norm(edge3);
    double angle2 = acos(dotProduct / (edge2Magnitude * edge3Magnitude)) * 180.0 / CV_PI;
    if (abs(angle2 - 90.0) > angleThreshold) {
    	//cout << "angle2: " << angle2 << endl;
    	return false;
    }

    // Calculate 3rd angle
    Point edge4 = point1 - point4;

    dotProduct = edge3.dot(edge4);
    double edge4Magnitude = norm(edge4);
    double angle3 = acos(dotProduct / (edge3Magnitude * edge4Magnitude)) * 180.0 / CV_PI;
    if (abs(angle3 - 90.0) > angleThreshold) {
    	//cout << "angle3: " << angle3 << endl;
    	return false;
    }

    // Calculate 4th angle
    dotProduct = edge4.dot(edge1);
    double angle4 = acos(dotProduct / (edge1Magnitude * edge4Magnitude)) * 180.0 / CV_PI;
    if (abs(angle4 - 90.0) > angleThreshold) {
    	//cout << "angle4: " << angle4 << endl;
    	return false;
    }

    return true;
}

double euclidean_distance(const Point2f& p1, const Point2f& p2)
{
    Point2f diff = p1 - p2;
    return norm(diff);
}


float matching(Element log, Element image, vector<DMatch> &inlier_matches, Mat &H){
	//Matching keypoints
	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
	vector<vector<DMatch>> knn_matches;
	matcher->knnMatch(log.descriptors, image.descriptors, knn_matches, 2);

	//Filtering matches using ratio
	vector<DMatch> good_matches;
	const float ratio_thresh = 0.75f;
	for (size_t i = 0; i < knn_matches.size(); i++){
		if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance){
			good_matches.push_back(knn_matches[i][0]);
		}
	}
	if (good_matches.size()<4) return 0;

	//Estimating a bounding box of object
	vector<Point2f> kp_log;
	vector<Point2f> kp_image;
	for(size_t i = 0; i < good_matches.size(); i++){
		kp_log.push_back(log.keypoints[good_matches[i].queryIdx].pt);
		kp_image.push_back(image.keypoints[good_matches[i].trainIdx].pt);
	}
	Mat inlierMask;
	H = findHomography(kp_log, kp_image, RANSAC, 3, inlierMask);
	//-- Get the corners from the image_1 (the object to be "detected")
	vector<Point2f> log_corners(4);
	log_corners[0] = Point2f(0, 0);
	log_corners[1] = Point2f((float)log.image.cols, 0);
	log_corners[2] = Point2f((float)log.image.cols, (float)log.image.rows);
	log_corners[3] = Point2f(0, (float)log.image.rows);
	vector<Point2f> image_corners(4);
	perspectiveTransform(log_corners, image_corners, H);

	//Checking the bounding box is a convex hull
	if (!isContourConvex(image_corners)) return 0;

	//Checking the bounding box is a nearly rectangle
	if (!isNearlyRectangle(image_corners,20)) {
		//cout << "reject by isNearlyRectangle" << endl;
		return 0;
	}

	//Filtering inlier matches
	for (int i = 0; i < inlierMask.rows; i++) {
        if (inlierMask.at<uchar>(i)) {
        	inlier_matches.push_back(good_matches[i]);
        }
    }

	return (float)inlier_matches.size()/log.descriptors.rows;
}

Mat draw_matches(Element log, Element image, vector<DMatch> good_matches, Mat H, float rs){
	//-- Draw matches
	Mat img_matches;
	drawMatches(log.image, log.keypoints, image.image, image.keypoints, good_matches, img_matches, Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

	int fontFace = FONT_HERSHEY_SIMPLEX;
    putText(img_matches, to_string(rs), Point(img_matches.cols/2,100), fontFace, 4, Scalar(0,255,0), 5);

    /*
    int targetWidth = 1000;
    int targetHeight = static_cast<int>((static_cast<double>(targetWidth) / img_matches.cols) * img_matches.rows);

    Mat outputImage;
    resize(img_matches, outputImage, cv::Size(targetWidth, targetHeight), 0, 0, cv::INTER_LINEAR);
	*/

	return img_matches;
}

int main(int argc, char* argv[]){
	cout << "Reading data..." << endl;
	read_logs();
	cout << nb_logs << " logs" << endl;
	cout << "Reading query image..." << endl;
	string image_name(argv[1]);
	Element image = read_image(image_name);
	vector<Mat> result_images;
	vector<float> rs;
	bool found=false;
	for(int l=0;l<nb_logs;l++){
		cout << "Matching with log " << l << " => ";
		vector<DMatch> good_matches;
		Mat H;
		float r = matching(logs[l], image, good_matches, H);
		if (r>=0.0005) {
			cout << "MATCHED" << endl;
			result_images.push_back(draw_matches(logs[l],image,good_matches, H, r));
			rs.push_back(r);
			found = true;
		} else {
			cout << "unmatched" << endl;
		}
	}

	if (found) {
		//sort based on rs
	    //std::iota(indices.begin(), indices.end(), 0);
	    std::vector<int> indices;
	    for (size_t i = 0; i < rs.size(); ++i) {
	        indices.push_back(i);
	    }

	    std::sort(indices.begin(), indices.end(), [&](int i, int j) {
	        return rs[i] > rs[j]; // Change to '<' if you want to sort in descending order
	    });

	    for (size_t i = 0; i < rs.size(); ++i) {
	        imwrite("rs_"+to_string(i+1)+".jpg",result_images[indices[i]]);
	    }
	    cout << rs.size() << " matched log(s)" << endl;
	}
	cout << "Finished" << endl;
	return 0;
}
