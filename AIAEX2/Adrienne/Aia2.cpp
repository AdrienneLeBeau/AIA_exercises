//============================================================================
// Name        : Aia2.cpp
// Author      : Ronny Haensch
// Version     : 1.0
// Copyright   : -
// Description : 
//============================================================================

#include "Aia2.h"

// calculates the contour line of all objects in an image
/*
img			the input image
objList		vector of contours, each represented by a two-channel matrix
thresh		threshold used to binarize the image
k			number of applications of the erosion operator
*/
void Aia2::getContourLine(const Mat& img, vector<Mat>& objList, int thresh, int k){
	int erosion_type = MORPH_ELLIPSE;
	int erosion_size = 0;
	int erosion_iterations = k;
	int const max_kernel_size = 21;


    // TO DO !!!
	/*
	Binary Image
	*/
	namedWindow("Binary Image", CV_WINDOW_AUTOSIZE);
	imshow("Binary Image", img);
	/*
	Threshold image
	*/	
	threshold(img, img, thresh, 255, THRESH_BINARY_INV);
	namedWindow("Threshold Demo", CV_WINDOW_AUTOSIZE);
	imshow("Threshold Demo", img);
	/*
	Erosion image
	*/
	namedWindow("Erosion Demo", CV_WINDOW_AUTOSIZE);	
	erode(img, img, Mat(), Point(-1, -1), k);
	imshow("Erosion Demo", img);
	/*
	find contours
	*/
	//findContours(img, objList, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);
	findContours(img, objList, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);
	

	
}

// calculates the (unnormalized!) fourier descriptor from a list of points
/*
contour		1xN 2-channel matrix, containing N points (x in first, y in second channel)
out		fourier descriptor (not normalized)
*/
Mat Aia2::makeFD(const Mat& contour){
 
    // TO DO !!!
	Mat contour_new;
	Mat contour_dft;
	contour.convertTo(contour_new, CV_32F);
	//dft(contour_new, contour_dft);
	dft(contour_new, contour_dft, DFT_COMPLEX_OUTPUT);
	
    return contour_dft;

}

// normalize a given fourier descriptor
/*
fd		the given fourier descriptor
n		number of used frequencies (should be even)
out		the normalized fourier descriptor
*/
Mat Aia2::normFD(const Mat& fd, int n){
   
  //plotFD(<???>, "fd not normalized", 0);
	plotFD(fd, "fd not normalized", 2000);
  
  // translation invariance
  // TO DO !!!
  //plotFD(<???>, "fd translation invariant", 0);
	Mat fd_translation_inv = fd.clone();
	fd_translation_inv.row(0) = 0;	
	plotFD(fd_translation_inv, "fd translation invariant", 2000);
  
  // scale invariance
  // TO DO !!!
	Mat fd_scale_inv = fd_translation_inv.clone();
	float f1 = abs(fd_scale_inv.at<float>(1, 1));
	fd_scale_inv / f1;
	plotFD(fd_scale_inv, "fd translation and scale invariant", 2000);
  
  // rotation invariance
  // TO DO !!!
  //plotFD(<???>, "fd translation, scale, and rotation invariant", 0);
	//Mat magnitude;
	//Mat angle;
	//cartToPolar(fd_new.row, fd_new.col, magnitude, angle);
	
  
  // smaller sensitivity for details
  // TO DO !!!
  //plotFD(<???>, "fd translation, scale, and rotation invariant, smaller sensitivity", 0);

  return fd;
}

// plot fourier descriptor
/*
fd	the fourier descriptor to be displayed
win	the window name
dur	wait number of ms or until key is pressed
*/
void Aia2::plotFD(const Mat& fd, string win, double dur){

   // TO DO !!!
	//Transform fourier-descriptor to pixel coordinates
	/*Mat fd_pixel;
	Mat fd_rot_normalized;
	Mat channel[2];
	split(fd, channel);
	channel[1] = Mat::zeros(fd.rows, fd.cols, CV_32F);													 
	merge(channel, 2, fd_rot_normalized);
	print(fd_rot_normalized);

	dft(fd_rot_normalized, fd_pixel, DFT_INVERSE);
	try
	{
		namedWindow(win, CV_WINDOW_AUTOSIZE);
		imshow(win, fd_pixel);
		waitKey(dur);
	}
	catch (cv::Exception & e)
	{
		cerr << e.msg << endl; // output exception message
	}
	*/
	Mat invDFT;
	dft(fd, invDFT, DFT_INVERSE); // Inverse transform to get image back.
								  //Mat Image. This is the base image matrix
	Mat Image(800, 800, CV_8UC1, Scalar(1));

	double maxVal;
	double minVal;
	double ch1_minVal;
	double ch1_maxVal;
	double ch2_minVal;
	double ch2_maxVal;
	// Split 1 dim 2 channel Mat to a vector of two 1 dim 1 chan Mat
	vector<Mat> CH_splitted;
	split(invDFT, CH_splitted);

	minMaxLoc(CH_splitted[0], &ch1_minVal, &ch1_maxVal);
	cout << ch1_maxVal << endl;
	cout << ch1_minVal << endl;

	minMaxLoc(CH_splitted[1], &ch2_minVal, &ch2_maxVal);
	cout << ch2_maxVal << endl;
	cout << ch2_minVal << endl;

	double X = max((ch1_maxVal - ch1_minVal), (ch2_maxVal - ch2_minVal));
	//
	CH_splitted[0].convertTo(CH_splitted[0], CV_32S, 1, -ch1_minVal);
	CH_splitted[1].convertTo(CH_splitted[1], CV_32S, 1, -ch2_minVal);
	CH_splitted[0].convertTo(CH_splitted[0], CV_32S, 600 / X, 100);
	CH_splitted[1].convertTo(CH_splitted[1], CV_32S, 600 / X, 100);

	//
	vector<Mat> big;
	Mat stContour;
	merge(CH_splitted, stContour);
	big.push_back(stContour);
	drawContours(Image, big, 0, 255, FILLED);
	namedWindow(win, CV_WINDOW_AUTOSIZE);
	imshow(win, Image);	
	waitKey(dur);
	//showImage(Image, win, dur);
	Image.convertTo(Image, CV_8U, 1, 0);
	string title = win + ".jpg";
	imwrite(title, Image);

	

    
}

/* *****************************
  GIVEN FUNCTIONS
***************************** */

// function loads input image, calls processing functions, and saves result
// in particular extracts FDs and compares them to templates
/*
img			path to query image
template1	path to template image of class 1
template2	path to template image of class 2
*/
void Aia2::run(string img, string template1, string template2){

	// process image data base
	// load image as gray-scale, paths in argv[2] and argv[3]
	Mat exC1 = imread( template1, 0);
	Mat exC2  = imread( template2, 0);
	if ( (!exC1.data) || (!exC2.data) ){
	    cout << "ERROR: Cannot load class examples in\n" << template1 << "\n" << template2 << endl;
	    cerr << "Press enter to continue..." << endl;
	    cin.get();
	    exit(-1);
	}

	// parameters
	// these two will be adjusted below for each image indiviudally
	int binThreshold;				// threshold for image binarization
	int numOfErosions;				// number of applications of the erosion operator
	// these two values work fine, but might be interesting for you to play around with them
	int steps = 32;					// number of dimensions of the FD
	double detThreshold = 0.01;		// threshold for detection

	// get contour line from images
	vector<Mat> contourLines1;
	vector<Mat> contourLines2;
	// TO DO !!!
	// --> Adjust threshold and number of erosion operations
	binThreshold = 135;
	numOfErosions = 2;
	getContourLine(exC1, contourLines1, binThreshold, numOfErosions);
	int mSize = 0, mc1 = 0, mc2 = 0, i = 0;
	for(vector<Mat>::iterator c = contourLines1.begin(); c != contourLines1.end(); c++,i++){
		if (mSize<c->rows){
			mSize = c->rows;
			mc1 = i;
		}
	}
	getContourLine(exC2, contourLines2, binThreshold, numOfErosions);
	for(vector<Mat>::iterator c = contourLines2.begin(); c != contourLines2.end(); c++, i++){
		if (mSize<c->rows){
			mSize = c->rows;
			mc2 = i;
		}
	}
	//waitKey();
	// calculate fourier descriptor
	Mat fd1 = makeFD(contourLines1.at(mc1));
	Mat fd2 = makeFD(contourLines2.at(mc2));

	// normalize  fourier descriptor
	Mat fd1_norm = normFD(fd1, steps);
	Mat fd2_norm = normFD(fd2, steps);

	// process query image
	// load image as gray-scale, path in argv[1]
	Mat query = imread( img, 0);
	if (!query.data){
	    cerr << "ERROR: Cannot load query image in\n" << img << endl;
	    cerr << "Press enter to continue..." << endl;
	    cin.get();
	    exit(-1);
	}
	
	// get contour lines from image
	vector<Mat> contourLines;
	// TO DO !!!
	// --> Adjust threshold and number of erosion operations
	binThreshold = 135;
	numOfErosions = 2;
	getContourLine(query, contourLines, binThreshold, numOfErosions);
	
	cout << "Found " << contourLines.size() << " object candidates" << endl;

	// just to visualize classification result
	Mat result(query.rows, query.cols, CV_8UC3);
	vector<Mat> tmp;
	tmp.push_back(query);
	tmp.push_back(query);
	tmp.push_back(query);
	merge(tmp, result);

	// loop through all contours found
	i = 1;
	for(vector<Mat>::iterator c = contourLines.begin(); c != contourLines.end(); c++, i++){

	    cout << "Checking object candidate no " << i << " :\t";
	  
		// color current object in yellow
	  	Vec3b col(0,255,255);
	    for(int p=0; p < c->rows; p++){
			result.at<Vec3b>(c->at<Vec2i>(p)[1], c->at<Vec2i>(p)[0]) = col;
	    }
	    showImage(result, "result", 0);
	    
	    // if fourier descriptor has too few components (too small contour), then skip it (and color it in blue)
	    if (c->rows < steps){
			cout << "Too less boundary points (" << c->rows << " instead of " << steps << ")" << endl;
			col = Vec3b(255,0,0);
	    }else{
			// calculate fourier descriptor
			Mat fd = makeFD(*c);
			// normalize fourier descriptor
			Mat fd_norm = normFD(fd, steps);
			// compare fourier descriptors
			double err1 = norm(fd_norm, fd1_norm)/steps;
			double err2 = norm(fd_norm, fd2_norm)/steps;
			// if similarity is too small, then reject (and color in cyan)
			if (min(err1, err2) > detThreshold){
				cout << "No class instance ( " << min(err1, err2) << " )" << endl;
				col = Vec3b(255,255,0);
			}else{
				// otherwise: assign color according to class
				if (err1 > err2){
					col = Vec3b(0,0,255);
					cout << "Class 2 ( " << err2 << " )" << endl;
				}else{
					col = Vec3b(0,255,0);
					cout << "Class 1 ( " << err1 << " )" << endl;
				}
			}
		}
		// draw detection result
	    for(int p=0; p < c->rows; p++){
			result.at<Vec3b>(c->at<Vec2i>(p)[1], c->at<Vec2i>(p)[0]) = col;
	    }
	    // for intermediate results, use the following line
	    showImage(result, "result", 0);

	}
	// save result
	imwrite("result.png", result);
	// show final result
	showImage(result, "result", 0);
}

// shows the image
/*
img	the image to be displayed
win	the window name
dur	wait number of ms or until key is pressed
*/
void Aia2::showImage(const Mat& img, string win, double dur){
  
    // use copy for normalization
    Mat tempDisplay = img.clone();
    if (img.channels() == 1) normalize(img, tempDisplay, 0, 255, CV_MINMAX);
    // create window and display omage
    namedWindow( win.c_str(), CV_WINDOW_AUTOSIZE );
    imshow( win.c_str(), tempDisplay );
    // wait
    if (dur>=0) waitKey(dur);
    
}

// function loads input image and calls processing function
// output is tested on "correctness" 
void Aia2::test(void){
	
	test_getContourLine();
	test_makeFD();
	//test_normFD();
	
}

void Aia2::test_getContourLine(void){

	vector<Mat> objList;
	Mat img(100, 100, CV_8UC1, Scalar(255));
	Mat roi(img, Rect(40,40,20,20));
	roi.setTo(0);
	getContourLine(img, objList, 128, 1);
	Mat cline(68,1,CV_32SC2);
	int k=0;
	for(int i=41; i<58; i++) cline.at<Vec2i>(k++) = Vec2i(41,i);
	for(int i=41; i<58; i++) cline.at<Vec2i>(k++) = Vec2i(i,58);
	for(int i=58; i>41; i--) cline.at<Vec2i>(k++) = Vec2i(58, i);
	for(int i=58; i>41; i--) cline.at<Vec2i>(k++) = Vec2i(i,41);
	if ( sum(cline != objList.at(0)).val[0] != 0 ){
		cout << "There might be a problem with Aia2::getContourLine(..)!" << endl;
		cin.get();
	}
}

void Aia2::test_makeFD(void){

	Mat cline(68,1,CV_32SC2);
	int k=0;
	for(int i=41; i<58; i++) cline.at<Vec2i>(k++) = Vec2i(41,i);
	for(int i=41; i<58; i++) cline.at<Vec2i>(k++) = Vec2i(i,58);
	for(int i=58; i>41; i--) cline.at<Vec2i>(k++) = Vec2i(58, i);
	for(int i=58; i>41; i--) cline.at<Vec2i>(k++) = Vec2i(i,41);
	
	Mat fd = makeFD(cline);
	if (fd.rows != cline.rows){
		cout << "There is be a problem with Aia2::makeFD(..):" << endl;
		cout << "\tThe number of frequencies does not match the number of contour points" << endl;
		cin.get();
		exit(-1);
	}
	if (fd.channels() != 2){
		cout << "There is be a problem with Aia2::makeFD(..):" << endl;
		cout << "\tThe fourier descriptor is supposed to be a two-channel, 1D matrix" << endl;
		cin.get();
		exit(-1);
	}
}

void Aia2::test_normFD(void){

	double eps = pow(10,-3);

	Mat cline(68,1,CV_32SC2);
	int k=0;
	for(int i=41; i<58; i++) cline.at<Vec2i>(k++) = Vec2i(41,i);
	for(int i=41; i<58; i++) cline.at<Vec2i>(k++) = Vec2i(i,58);
	for(int i=58; i>41; i--) cline.at<Vec2i>(k++) = Vec2i(58, i);
	for(int i=58; i>41; i--) cline.at<Vec2i>(k++) = Vec2i(i,41);
	
	Mat fd = makeFD(cline);
	Mat nfd = normFD(fd, 32);
	if (nfd.channels() != 1){
		cout << "There is be a problem with Aia2::normFD(..):" << endl;
		cout << "\tThe normalized fourier descriptor is supposed to be a one-channel, 1D matrix" << endl;
		cin.get();
		exit(-1);
	}
	if (abs(nfd.at<float>(0)) > eps){
		cout << "There is be a problem with Aia2::normFD(..):" << endl;
		cout << "\tThe F(0)-component of the normalized fourier descriptor F is supposed to be 0" << endl;
		cin.get();
		exit(-1);
	}
	if ((abs(nfd.at<float>(1)-1.) > eps) && (abs(nfd.at<float>(nfd.rows-1)-1.) > eps)){
		cout << "There is be a problem with Aia2::normFD(..):" << endl;
		cout << "\tThe F(1)-component of the normalized fourier descriptor F is supposed to be 1" << endl;
		cout << "\tBut what if the unnormalized F(1)=0?" << endl;
		cin.get();
		exit(-1);
	}
	if (nfd.rows != 32){
		cout << "There is be a problem with Aia2::normFD(..):" << endl;
		cout << "\tThe number of components does not match the specified number of components" << endl;
		cin.get();
		exit(-1);
	}
}
