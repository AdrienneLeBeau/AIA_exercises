//============================================================================
// Name        : Aia3.cpp
// Author      : Ronny Haensch
// Version     : 1.0
// Copyright   : -
// Description :
//============================================================================
/*
//g++ -ggdb `pkg-config --cflags --libs opencv` main.cpp Aia3.cpp -o out
./out
./out img/moneyTemplate100.jpg
*/
#include "Aia3.h"

// shows hough space, eg. as a projection of angle- and scale-dimensions down to a single image
/*
  houghSpace:	the hough space as generated by generalHough(..)
*/
void Aia3::plotHough(const vector< vector<Mat> >& houghSpace){
/*
int numberofs = houghSpace.size();
int numberoftheta = houghSpace[0].size();
int nrowsmat = houghSpace[0][0].rows;
int ncolsmat = houghSpace[0][0].cols;

cout << "s = " << numberofs << endl;
cout << "theta  = " << numberoftheta << endl;

Mat Houghsumma = Mat::zeros(nrowsmat,ncolsmat,CV_32FC1);
//cout << "rows" << houghSpace.rows << "cols" << houghSpace.cols << endl;
for(int i = 0; i < numberofs; i++){
  for(int j = 0; j < numberoftheta; j++){

  //Houghsumma.add(houghSpace[i][j]);// = Houghsumma + houghSpace[i][j];//.clone();
  Houghsumma+=houghSpace[i][j].clone();
  //cout << houghSpace[1][1] << endl;
  }
}
double min, max;
minMaxLoc(Houghsumma, &min, &max);
Houghsumma.convertTo(Houghsumma,CV_32FC1,255/max,0);
//cout << Houghsumma << endl;

//cout << "theta  = " << Houghsumma << endl;

showImage(Houghsumma, "summan", 0);
*/
}

// creates the fourier-spectrum of the scaled and rotated template
/*
  templ:	the object template; binary image in templ[0], complex gradient in templ[1]
  scale:	the scale factor to scale the template
  angle:	the angle to rotate the template
  fftMask:	the generated fourier-spectrum of the template (initialized outside of this function)
*/
void Aia3::makeFFTObjectMask(const vector<Mat>& templ, double scale, double angle, Mat& fftMask){
  /*
  Rotate and scale image
  */
  Mat Binarytempl = rotateAndScale(templ[0], angle, scale);
  Mat Gradtempl = rotateAndScale(templ[1], angle, scale);
  /*
  Create binary edge mask
  */
  // Split x-y channels to two mat elements in a vector
  vector<Mat> splittedmarix;
  split(Gradtempl,splittedmarix);
  Mat Dx = splittedmarix[0];
  Mat Dy = splittedmarix[1];


  float COS = cos(-angle); // OBS
  float SIN = sin(-angle); // OBS

  Mat Cx = Dx.mul(Binarytempl)*COS - Dy.mul(Binarytempl)*SIN;
  Mat Cy = Dy.mul(Binarytempl)*COS + Dx.mul(Binarytempl)*SIN;
  // We dont need to merge to 2 channels yet! We do it just before send to dft
  /*
  Normalize gradients so that sum is unit
  */
  Mat xpowmatrix;
  Mat ypowmatrix;
  Mat OXY;
  pow(Cx, 2, xpowmatrix);
  pow(Cy, 2, ypowmatrix);
  Mat sumpowmatrix = xpowmatrix + ypowmatrix;
  pow(sumpowmatrix,0.5,OXY);
  Scalar Norm = sum(OXY);
  Cx /= Norm[0];
  Cy /= Norm[0];
  /*
  Copy to larger matrix to get same dimensions as image to be searched in
  MaskCH1: gradient masked to binary edge map and copied to a matrix of right size, real part
  MaskCH2: gradient masked to binary edge map and copied to a matrix of right size, imag part
  */
  Mat MaskCH1 = Mat::zeros(fftMask.rows, fftMask.cols, CV_32FC1);
  Mat MaskCH2 = Mat::zeros(fftMask.rows, fftMask.cols, CV_32FC1);

  int Obj_rows = Cx.rows;
  int Obj_cols = Cx.cols;

  for(int i = 0; i < Obj_rows; i++){
    for(int j = 0; j < Obj_cols; j++){
      MaskCH1.at<float>(i,j) =  Cx.at<float>(i,j);
      MaskCH2.at<float>(i,j) =  Cy.at<float>(i,j);
    }
  }
  // Merge because circShift wants merged image
  vector<Mat> splittedmatrix2;
  Mat mergedMask;
  splittedmatrix2.push_back(MaskCH1);
  splittedmatrix2.push_back(MaskCH2);
  merge(splittedmatrix2,mergedMask);
  /*
  Translate filter center to origin of big image
  // ...But what is object center? mass-center? or just x.rows/2, y.rows/2?
  */
  // Descide how much to shift
  int dx = - Obj_cols/2;
  int dy = - Obj_rows/2;
  Mat Mask_centered = circShift(mergedMask,dx,dy);
  // Split image again
  split(Mask_centered,splittedmatrix2);
  /*
  showImage(splittedmatrix2[0],"chan1",0);
  showImage(splittedmatrix2[1],"chan2",0);
  */
  dft(Mask_centered,fftMask);
  //fftMask.row(0).setTo(0);
  //fftMask.col(0).setTo(0);
  //cout << "FFTMASK" << fftMask << endl;
  //waitKey(0);

}

// computes the hough space of the general hough transform
/*
  gradImage:	the gradient image of the test image
  templ:		the template consisting of binary image and complex-valued directional gradient image
  scaleSteps:	scale resolution
  scaleRange:	range of investigated scales [min, max]
  angleSteps:	angle resolution
  angleRange:	range of investigated angles [min, max)
  return:		the hough space: outer vector over scales, inner vector of angles
*/
vector< vector<Mat> > Aia3::generalHough(const Mat& gradImage, const vector<Mat>& templ, double scaleSteps, double* scaleRange, double angleSteps, double* angleRange){
vector< vector<Mat> > hough(int(scaleSteps),vector<Mat>(int(angleSteps),Mat(gradImage.rows,gradImage.cols,CV_32FC1)));
// Initialize a mat of right size to pass to makeFFTObjectMask
Mat fftMask = Mat::zeros(gradImage.rows,gradImage.cols,CV_32FC2);

Mat Imagfft;
dft(gradImage, Imagfft);

float delta_s = (scaleRange[1] - scaleRange[0])/scaleSteps;
float delta_theta = (angleRange[1] - angleRange[0])/angleSteps;
cout << "2" << endl;

for(int s = 0; s < scaleSteps; s++){
    float s_loop = scaleRange[0] + s * delta_s;
    for( int theta = 0; theta < angleSteps; theta++){
        float theta_loop = angleRange[0] + theta * delta_theta;

        //makeFFTObjectMask(MaskCH1_centered, s_loop, theta_loop, fftMask);
        makeFFTObjectMask(templ, s_loop, theta_loop , fftMask);

        // Now to be able to muliply them we have to divide into 2 channels
        vector <Mat> Imagfftsplit;
        vector <Mat> fftMasksplit;

        split(Imagfft,Imagfftsplit);
        split(fftMask,fftMasksplit);

        // Mulitiply in Frequencydomain -> correlation in time domain.
        // OBS CONJUGATE-MULTIPLICATION
        Mat PartReal = Imagfftsplit[0].mul(fftMasksplit[0]) + Imagfftsplit[1].mul(fftMasksplit[1]);
        Mat PartImag = Imagfftsplit[1].mul(fftMasksplit[0]) - Imagfftsplit[0].mul(fftMasksplit[1]);

        vector <Mat> freqmultiplication;
        freqmultiplication.push_back(PartReal);
        freqmultiplication.push_back(PartImag);

        Mat freqmultiplication2;
        merge(freqmultiplication,freqmultiplication2);
        //cout << "keyboard" << endl;

        Mat Correlationresult;
        dft(freqmultiplication2, Correlationresult, DFT_INVERSE);

        vector <Mat> splittedmatrix3;
        split(Correlationresult,splittedmatrix3);

        pow(splittedmatrix3[0],2,splittedmatrix3[0]);
        pow(splittedmatrix3[0],0.5,splittedmatrix3[0]);
        //showImage(splittedmatrix3[0],"correlation channel 1",0);
        //showImage(splittedmatrix3[1],"correlation channel 2",0);
        hough[s][theta] = splittedmatrix3[0].clone();
        //hough[theta][s] = splittedmatrix3[0].clone();

        }

    }

return hough;
}

// creates object template from template image
/*
  templateImage:	the template image
  sigma:			standard deviation of directional gradient kernel
  templateThresh:	threshold for binarization of the template image
  return:			the computed template
*/
vector<Mat> Aia3::makeObjectTemplate(const Mat& templateImage, double sigma, double templateThresh){

     //Mat splittedmarix;
     vector<Mat> splittedmarix;
     Mat gradimage = calcDirectionalGrad(templateImage, sigma);
     split(gradimage,splittedmarix);

     //cout << splittedmarix[0] << endl;
     //cout << splittedmarix[1] << endl;

     Mat absval;
     absval.zeros(gradimage.cols,gradimage.cols,CV_64F);

     Mat xpowmatrix;
     Mat ypowmatrix;
     Mat OXY;

     pow(splittedmarix[0], 2, xpowmatrix);
     pow(splittedmarix[1], 2, ypowmatrix);

     Mat sumpowmatrix = xpowmatrix + ypowmatrix;
     pow(sumpowmatrix,0.5,OXY);

     double min, max;
     minMaxLoc(OXY, &min, &max);
     OXY.setTo(0, OXY < max*templateThresh);
     OXY.setTo(255, OXY > 0);

     //showImage(OXY,"Binary image indicating pixels", 0);
     //waitKey(0);

     vector <Mat> returnvector;
     returnvector.push_back(OXY.clone());
     returnvector.push_back(gradimage.clone());

     return returnvector;
     cout << "1" << endl;

}

/* *****************************
  GIVEN FUNCTIONS
***************************** */

// loads template and test images, sets parameters and calls processing routine
/*
tmplImg:	path to template image
testImg:	path to test image
*/
void Aia3::run(string tmplImg, string testImg){

    // processing parameter
    double sigma 			= 1;		// standard deviation of directional gradient kernel
    double templateThresh 	= 0.3;		// relative threshold for binarization of the template image
    // TO DO !!!
    // ****
	// Set parameters to reasonable values
    double objThresh 		= 0.6;		// relative threshold for maxima in hough space
    double scaleSteps 		= 50;		// scale resolution in terms of number of scales to be investigated
    double scaleRange[2];				// scale of angles [min, max]
	scaleRange[0] 			= 0.5;
	scaleRange[1] 			= 2;
    double angleSteps 		= 50;		// angle resolution in terms of number of angles to be investigated
    double angleRange[2];				// range of angles [min, max)
  angleRange[0] 			= 0;
	angleRange[1] 			= 2*CV_PI;
	// ****

	Mat params = (Mat_<float>(1,9) << sigma, templateThresh, objThresh, scaleSteps, scaleRange[0], scaleRange[1], angleSteps, angleRange[0], angleRange[1]);

    // load template image as gray-scale, paths in argv[1]
    Mat templateImage = imread( tmplImg, 0);
    if (!templateImage.data){
		cerr << "ERROR: Cannot load template image from\n" << tmplImg << endl;
	    cerr << "Press enter..." << endl;
	    cin.get();
		exit(-1);
    }
    // convert 8U to 32F
    templateImage.convertTo(templateImage, CV_32FC1);
    // show template image
    showImage(templateImage, "Template image", 0);

    // load test image
    Mat testImage = imread( testImg, 0);
	if (!testImage.data){
		cerr << "ERROR: Cannot load test image from\n" << testImg << endl;
	    cerr << "Press enter..." << endl;
	    cin.get();
		exit(-1);
	}
	// and convert it from 8U to 32F
	testImage.convertTo(testImage, CV_32FC1);
    // show test image
    showImage(testImage, "testImage", 0);

    // start processing
    process(templateImage, testImage, params);

}

// loads template and create test image, sets parameters and calls processing routine
/*
tmplImg:	path to template image
angle:		rotation angle in degree of the test object
scale:		scale of the test object
*/
void Aia3::test(string tmplImg, float angle, float scale){

	// angle to rotate template image (in radian)
	double testAngle = angle/180.*CV_PI;
	// scale to scale template image
	double testScale = scale;

    // processing parameter
    double sigma 			= 1;		// standard deviation of directional gradient kernel
    double templateThresh 	= 0.7;		// relative threshold for binarization of the template image
    double objThresh		= 0.85;		// relative threshold for maxima in hough space
    double scaleSteps 		= 20;		// scale resolution in terms of number of scales to be investigated
    double scaleRange[2];				// scale of angles [min, max]
	scaleRange[0] 			= 1;
	scaleRange[1] 			= 2;
    double angleSteps 		= 40;		// angle resolution in terms of number of angles to be investigated
    double angleRange[2];				// range of angles [min, max)
	angleRange[0] 			= 0;
	angleRange[1] 			= 2*CV_PI;

	Mat params = (Mat_<float>(1,9) << sigma, templateThresh, objThresh, scaleSteps, scaleRange[0], scaleRange[1], angleSteps, angleRange[0], angleRange[1]);

    // load template image as gray-scale, paths in argv[1]
    Mat templateImage = imread( tmplImg, 0);
    if (!templateImage.data){
		cerr << "ERROR: Cannot load template image from\n" << tmplImg << endl;
		cerr << "Press enter..." << endl;
	    cin.get();
		exit(-1);
    }
    // convert 8U to 32F
    templateImage.convertTo(templateImage, CV_32FC1);
    // show template image
    showImage(templateImage, "Template Image", 0);

    // generate test image
    Mat testImage = makeTestImage(templateImage, testAngle, testScale, scaleRange);
    // show test image
    showImage(testImage, "Test Image", 0);

	// start processing
    process(templateImage, testImage, params);
}

void Aia3::process(const Mat& templateImage, const Mat& testImage, const Mat& params){

	// processing parameter
    double sigma			= params.at<float>(0);		// standard deviation of directional gradient kernel
    double templateThresh 	= params.at<float>(1);		// relative threshold for binarization of the template image
    double objThresh 		= params.at<float>(2);		// relative threshold for maxima in hough space
    double scaleSteps 		= params.at<float>(3);		// scale resolution in terms of number of scales to be investigated
    double scaleRange[2];								// scale of angles [min, max]
	scaleRange[0] 			= params.at<float>(4);
	scaleRange[1] 			= params.at<float>(5);
    double angleSteps 		= params.at<float>(6);		// angle resolution in terms of number of angles to be investigated
	double angleRange[2];								// range of angles [min, max)
    angleRange[0] 			= params.at<float>(7);
	angleRange[1] 			= params.at<float>(8);

	// calculate directional gradient of test image as complex numbers (two channel image)
    Mat gradImage = calcDirectionalGrad(testImage, sigma);

    // generate template from template image
    // templ[0] == binary image
    // templ[0] == directional gradient image
    vector<Mat> templ = makeObjectTemplate(templateImage, sigma, templateThresh);

    // show binary image
    showImage(templ[0], "Binary part of template", 0);

    // perfrom general hough transformation
    vector< vector<Mat> > houghSpace = generalHough(gradImage, templ, scaleSteps, scaleRange, angleSteps, angleRange);
	// plot hough space (max over angle- and scale-dimension)
    plotHough(houghSpace);
    // find maxima in hough space
    vector<Scalar> objList;
    findHoughMaxima(houghSpace, objThresh, objList);

    // print found objects on screen
    cout << "Number of objects: " << objList.size() << endl;
    int i=0;
		for(vector<Scalar>::const_iterator it = objList.begin(); it != objList.end(); it++, i++){
		cout << i << "\tScale:\t" << (scaleRange[1] - scaleRange[0])/(scaleSteps-1)*(*it).val[0] + scaleRange[0];
		cout << "\tAngle:\t" << ((angleRange[1] - angleRange[0])/(angleSteps)*(*it).val[1] + angleRange[0])/CV_PI*180;
		cout << "\tPosition:\t(" << (*it).val[2] << ", " << (*it).val[3] << " )" << endl;
    }

    // show final detection result
    plotHoughDetectionResult(testImage, templ, objList, scaleSteps, scaleRange, angleSteps, angleRange);

}
// computes directional gradients
/*
  image:	the input image
  sigma:	standard deviation of the kernel
  return:	the two-channel gradient image
*/
Mat Aia3::calcDirectionalGrad(const Mat& image, double sigma){

  // compute kernel size
  int ksize = max(sigma*3,3.);
  if (ksize % 2 == 0)  ksize++;
  double mu = ksize/2.0;

  // generate kernels for x- and y-direction
  double val, sum=0;
  Mat kernel(ksize, ksize, CV_32FC1);
  //Mat kernel_y(ksize, ksize, CV_32FC1);
  for(int i=0; i<ksize; i++){
      for(int j=0; j<ksize; j++){
		val  = pow((i+0.5-mu)/sigma,2);
		val += pow((j+0.5-mu)/sigma,2);
		val = exp(-0.5*val);
		sum += val;
		kernel.at<float>(i, j) = -(j+0.5-mu)*val;
     }
  }
  kernel /= sum;
  // use those kernels to compute gradient in x- and y-direction independently
  vector<Mat> grad(2);
  filter2D(image, grad[0], -1, kernel);
  filter2D(image, grad[1], -1, kernel.t());
  // combine both real-valued gradient images to a single complex-valued image
  Mat output;
  merge(grad, output);

  return output;
}

// rotates and scales a given image
/*
  image:	the image to be scaled and rotated
  angle:	rotation angle in radians
  scale:	scaling factor
  return:	transformed image
*/
Mat Aia3::rotateAndScale(const Mat& image, double angle, double scale){

    // create transformation matrices
    // translation to origin
    Mat T = Mat::eye(3, 3, CV_32FC1);
    T.at<float>(0, 2) = -image.cols/2.0;
    T.at<float>(1, 2) = -image.rows/2.0;
    // rotation
    Mat R = Mat::eye(3, 3, CV_32FC1);
    R.at<float>(0, 0) =  cos(angle);
    R.at<float>(0, 1) = -sin(angle);
    R.at<float>(1, 0) =  sin(angle);
    R.at<float>(1, 1) =  cos(angle);
    // scale
    Mat S = Mat::eye(3, 3, CV_32FC1);
    S.at<float>(0, 0) = scale;
    S.at<float>(1, 1) = scale;
    // combine
    Mat H = R*S*T;

    // compute corners of warped image
    Mat corners(1, 4, CV_32FC2);
    corners.at<Vec2f>(0, 0) = Vec2f(0,0);
    corners.at<Vec2f>(0, 1) = Vec2f(0,image.rows);
    corners.at<Vec2f>(0, 2) = Vec2f(image.cols,0);
    corners.at<Vec2f>(0, 3) = Vec2f(image.cols,image.rows);
    perspectiveTransform(corners, corners, H);

    // compute size of resulting image and allocate memory
    float x_start = min( min( corners.at<Vec2f>(0, 0)[0], corners.at<Vec2f>(0, 1)[0]), min( corners.at<Vec2f>(0, 2)[0], corners.at<Vec2f>(0, 3)[0]) );
    float x_end   = max( max( corners.at<Vec2f>(0, 0)[0], corners.at<Vec2f>(0, 1)[0]), max( corners.at<Vec2f>(0, 2)[0], corners.at<Vec2f>(0, 3)[0]) );
    float y_start = min( min( corners.at<Vec2f>(0, 0)[1], corners.at<Vec2f>(0, 1)[1]), min( corners.at<Vec2f>(0, 2)[1], corners.at<Vec2f>(0, 3)[1]) );
    float y_end   = max( max( corners.at<Vec2f>(0, 0)[1], corners.at<Vec2f>(0, 1)[1]), max( corners.at<Vec2f>(0, 2)[1], corners.at<Vec2f>(0, 3)[1]) );

    // create translation matrix in order to copy new object to image center
    T.at<float>(0, 0) = 1;
    T.at<float>(1, 1) = 1;
    T.at<float>(2, 2) = 1;
    T.at<float>(0, 2) = (x_end - x_start + 1)/2.0;
    T.at<float>(1, 2) = (y_end - y_start + 1)/2.0;

    // change homography to take necessary translation into account
    H = T * H;
    // warp image and copy it to output image
    Mat output;
    warpPerspective(image, output, H, Size(x_end - x_start + 1, y_end - y_start + 1), CV_INTER_LINEAR);

    return output;

}

// generates the test image as a transformed version of the template image
/*
  temp:		the template image
  angle:	rotation angle
  scale:	scaling factor
  scaleRange:	scale range [min,max], used to determine the image size
*/
Mat Aia3::makeTestImage(const Mat& temp, double angle, double scale, double* scaleRange){

    // rotate and scale template image
    Mat small = rotateAndScale(temp, angle, scale);

    // create empty test image
    Mat testImage = Mat::zeros(temp.rows*scaleRange[1]*2, temp.cols*scaleRange[1]*2, CV_32FC1);
    // copy new object into test image
    Mat tmp;
    Rect roi;
    roi = Rect( (testImage.cols - small.cols)*0.5, (testImage.rows - small.rows)*0.5, small.cols, small.rows);
    tmp = Mat(testImage, roi);
    small.copyTo(tmp);

    return testImage;
}

// shows the detection result of the hough transformation
/*
  testImage:	the test image, where objects were searched (and hopefully found)
  templ:		the template consisting of binary image and complex-valued directional gradient image
  objList:		list of objects as defined by findHoughMaxima(..)
  scaleSteps:	scale resolution
  scaleRange:	range of investigated scales [min, max]
  angleSteps:	angle resolution
  angleRange:	range of investigated angles [min, max)
*/
void Aia3::plotHoughDetectionResult(const Mat& testImage, const vector<Mat>& templ, const vector<Scalar>& objList, double scaleSteps, double* scaleRange, double angleSteps, double* angleRange){

    // some matrices to deal with color
    Mat red = testImage.clone();
    Mat green = testImage.clone();
    Mat blue = testImage.clone();
    Mat tmp = Mat::zeros(testImage.rows, testImage.cols, CV_32FC1);

    // scale and angle of current object
    double scale, angle;

    // for all objects
    for(vector<Scalar>::const_iterator it = objList.begin(); it != objList.end(); it++){
		// compute scale and angle of current object
		scale = (scaleRange[1] - scaleRange[0])/(scaleSteps-1)*(*it).val[0] + scaleRange[0];
		angle = ((angleRange[1] - angleRange[0])/(angleSteps)*(*it).val[1] + angleRange[0]);

		// use scale and angle in order to generate new binary mask of template
		Mat binMask = rotateAndScale(templ[0], angle, scale);

		// perform boundary checks
		Rect binArea = Rect(0, 0, binMask.cols, binMask.rows);
		Rect imgArea = Rect((*it).val[2]-binMask.cols/2., (*it).val[3]-binMask.rows/2, binMask.cols, binMask.rows);
		if ( (*it).val[2]-binMask.cols/2 < 0 ){
			binArea.x = abs( (*it).val[2]-binMask.cols/2 );
			binArea.width = binMask.cols - binArea.x;
			imgArea.x = 0;
			imgArea.width = binArea.width;
		}
		if ( (*it).val[3]-binMask.rows/2 < 0 ){
			binArea.y = abs( (*it).val[3]-binMask.rows/2 );
			binArea.height = binMask.rows - binArea.y;
			imgArea.y = 0;
			imgArea.height = binArea.height;
		}
		if ( (*it).val[2]-binMask.cols/2 + binMask.cols >= tmp.cols ){
			binArea.width = binMask.cols - ( (*it).val[2]-binMask.cols/2 + binMask.cols - tmp.cols );
			imgArea.width = binArea.width;
		}
		if ( (*it).val[3]-binMask.rows/2 + binMask.rows >= tmp.rows ){
			binArea.height = binMask.rows - ( (*it).val[3]-binMask.rows/2 + binMask.rows - tmp.rows );
			imgArea.height = binArea.height;
		}
		// copy this object instance in new image of correct size
		tmp.setTo(0);
		Mat binRoi = Mat(binMask, binArea);
		Mat imgRoi = Mat(tmp, imgArea);
		binRoi.copyTo(imgRoi);

		// delete found object from original image in order to reset pixel values with red (which are white up until now)
		binMask = 1 - binMask;
		imgRoi = Mat(red, imgArea);
		multiply(imgRoi, binRoi, imgRoi);
		imgRoi = Mat(green, imgArea);
		multiply(imgRoi, binRoi, imgRoi);
		imgRoi = Mat(blue, imgArea);
		multiply(imgRoi, binRoi, imgRoi);

		// change red channel
		red = red + tmp*255;
    }
    // generate color image
    vector<Mat> color;
    color.push_back(blue);
    color.push_back(green);
    color.push_back(red);
    Mat display;
    merge(color, display);
    // display color image
    showImage(display, "result", 0);
    // save color image
    imwrite("detectionResult.png", display);
}

// seeks for local maxima within the hough space
/*
  a local maxima has to be larger than all its 8 spatial neighbors, as well as the largest value at this position for all scales and orientations
  houghSpace:	the computed hough space
  objThresh:	relative threshold for maxima in hough space
  objList:	list of detected objects
*/
void Aia3::findHoughMaxima(const vector< vector<Mat> >& houghSpace, double objThresh, vector<Scalar>& objList){

    // get maxima over scales and angles
    Mat maxImage = Mat::zeros(houghSpace.at(0).at(0).rows, houghSpace.at(0).at(0).cols, CV_32FC1 );
    for(vector< vector<Mat> >::const_iterator it = houghSpace.begin(); it != houghSpace.end(); it++){
	for(vector<Mat>::const_iterator img = (*it).begin(); img != (*it).end(); img++){
	    max(*img, maxImage, maxImage);
	}
    }
    // get global maxima
    double min, max;
    minMaxLoc(maxImage, &min, &max);

    // define threshold
    double threshold = objThresh * max;

    // spatial non-maxima suppression
    Mat bin = Mat(houghSpace.at(0).at(0).rows, houghSpace.at(0).at(0).cols, CV_32FC1, -1);
    for(int y=0; y<maxImage.rows; y++){
		for(int x=0; x<maxImage.cols; x++){
			// init
			bool localMax = true;
			// check neighbors
			for(int i=-1; i<=1; i++){
				int new_y = y + i;
				if ((new_y < 0) or (new_y >= maxImage.rows)){
					continue;
				}
				for(int j=-1; j<=1; j++){
					int new_x = x + j;
					if ((new_x < 0) or (new_x >= maxImage.cols)){
					continue;
					}
					if (maxImage.at<float>(new_y, new_x) > maxImage.at<float>(y, x)){
					localMax = false;
					break;
					}
				}
				if (!localMax)
					break;
			}
			// check if local max is larger than threshold
			if ( (localMax) and (maxImage.at<float>(y, x) > threshold) ){
				bin.at<float>(y, x) = maxImage.at<float>(y, x);
			}
		}
    }

    // loop through hough space after non-max suppression and add objects to object list
    double scale, angle;
    scale = 0;
    for(vector< vector<Mat> >::const_iterator it = houghSpace.begin(); it != houghSpace.end(); it++, scale++){
		angle = 0;
		for(vector<Mat>::const_iterator img = (*it).begin(); img != (*it).end(); img++, angle++){
			for(int y=0; y<bin.rows; y++){
				for(int x=0; x<bin.cols; x++){
					if ( (*img).at<float>(y, x) == bin.at<float>(y, x) ){
					// create object list entry consisting of scale, angle, and position where object was detected
					Scalar cur;
					cur.val[0] = scale;
					cur.val[1] = angle;
					cur.val[2] = x;
					cur.val[3] = y;
					objList.push_back(cur);
					}
				}
			}
		}
    }
}

// shows the image
/*
img:	the image to be displayed
win:	the window name
dur:	wait number of ms or until key is pressed
*/
void Aia3::showImage(const Mat& img, string win, double dur){

    // use copy for normalization
    Mat tempDisplay;
    if (img.channels() == 1)
	normalize(img, tempDisplay, 0, 255, CV_MINMAX);
    else
	tempDisplay = img.clone();

    tempDisplay.convertTo(tempDisplay, CV_8UC1);

    // create window and display omage
    namedWindow( win.c_str(), 0 );
    imshow( win.c_str(), tempDisplay );
    // wait
    if (dur>=0) cvWaitKey(dur);
    // be tidy
    destroyWindow(win.c_str());

}

// Performes a circular shift in (dx,dy) direction
/*
in:		input matrix
out:	circular shifted matrix
dx:		shift in x-direction
dy:		shift in y-direction
*/
Mat Aia3::circShift(const Mat& in, int dx, int dy){

	Mat tmp = Mat::zeros(in.rows, in.cols, in.type());

	int x, y, new_x, new_y;

	for(y=0; y<in.rows; y++){

	      // calulate new y-coordinate
	      new_y = y + dy;
	      if (new_y<0)
		  new_y = new_y + in.rows;
	      if (new_y>=in.rows)
		  new_y = new_y - in.rows;

	      for(x=0; x<in.cols; x++){

		  // calculate new x-coordinate
		  new_x = x + dx;
		  if (new_x<0)
			new_x = new_x + in.cols;
		  if (new_x>=in.cols)
			new_x = new_x - in.cols;

		  tmp.at<Vec2f>(new_y, new_x) = in.at<Vec2f>(y, x);

	    }
	}
	return tmp;
}
