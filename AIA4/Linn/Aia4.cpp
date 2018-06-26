//========================================================
// Name        : Aia5.cpp
// Author      : Ronny Haensch
// Version     : 1.0
// Copyright   : -
// Description :
//============================================================================

#include "Aia4.h"

// Computes the log-likelihood of each feature vector in every component of the supplied mixture model.
/*
model:     	structure containing model parameters
features:  	matrix of feature vectors
return: 	the log-likelihood log(p(x_i|y_i=j))
*/
Mat Aia5::calcCompLogL(const vector<struct comp*>& model, const Mat& features){

	int C_current = model.size();	// Number of current clusters
	int Number_features = features.cols;	// Number of features/samples (x)
	int Number_dimensions = features.rows;	// Number of dimension of the vectors (x_i)

	// Creating a Mat containing the probabilities log(p(x_i|mu_c,Cov_c)) that are to be
	// the output.
	Mat log_P_out = Mat::zeros(C_current,Number_features,CV_32FC1);
	float Z = -Number_dimensions/2*log(2*CV_PI);
	Mat Ones = Mat::ones(Number_dimensions,Number_features,CV_32FC1);

	for(int i = 0; i < C_current; i++){

		// Calculating the inverse of Cov-matrix
		Mat inv_M;
		invert(model.at(i)->covar,inv_M);
		float Constant = -0.5*log(determinant(model.at(i)->covar));

			for(int j = 0; j < Number_features; j++){

				Mat vec_diff = features.col(j)-model.at(i)->mean;	// Defining the difference vector (x_i-y_c)
				Mat Matrix_mul = -0.5*(vec_diff.t()*inv_M*vec_diff) +  Constant - Z; // OBS MAYBE NEED TO ADD Z, Maybe not (Normalize)

				log_P_out.at<float>(i,j) = Matrix_mul.at<float>(0,0);

			}
		}
		/*
	Mat img0(600,600,CV_8UC3);
	double low = -500.0;
	double high = +500.0;
	randu(img0, Scalar(low), Scalar(high));

	Mat cm_img0;

	applyColorMap(img0, cm_img0, COLORMAP_JET);
	// Show the result:
	imshow("cm_img0", cm_img0);
	waitKey(0);
	*/
	return log_P_out;

}

// Computes the log-likelihood of each feature by combining the likelihoods in all model components.
/*
model:     structure containing model parameters
features:  matrix of feature vectors
return:	   the log-likelihood of feature number i in the mixture model (the log of the summation of alpha_j p(x_i|y_i=j) over j)
*/

Mat Aia5::calcMixtureLogL(const vector<struct comp*>& model, const Mat& features){

	int C_current = model.size();	// Number of current clusters
	int Number_features = features.cols;	// Number of features/samples (x)
	int Number_dimensions = features.rows;	// Number of dimension of the vectors (x_i)

	// Getting the log-probabilities from the calcCompLogL function
  Mat log_Prob_all_Cs = calcCompLogL( model, features);

	// Creating the Mat that will contain the log-likelihood of each feature by
	// combining the likelihoods in all model components
	Mat log_sum = Mat::zeros(1,Number_features, CV_32FC1);

	for(int j = 0; j < Number_features; j++){
		// OBS maybe need to scale!!
		//double min, scale;
		//minMaxLoc(Prob.col(j), &min, &scale);
		float exp_sum = 0;

			for(int i = 0; i < C_current; i++){
				//cout << (model.at(i)->weight)*exp(log_Prob_all_Cs.at<float>(i,j)) << endl;
				exp_sum += (model.at(i)->weight)*exp(log_Prob_all_Cs.at<float>(i,j));

			}

		log_sum.at<float>(0,j) = log(exp_sum);

		}

	return log_sum;
}

// Computes the posterior over components (the degree of component membership) for each feature.
/*
model:     	structure containing model parameters
features:  	matrix of feature vectors
return:		the posterior p(y_i=j|x_i)
*/
Mat Aia5::gmmEStep(const vector<struct comp*>& model, const Mat& features){

	int C_current = model.size();	// Number of current clusters
	int Number_features = features.cols;	// Number of features/samples (x)
	int Number_dimensions = features.rows;	// Number of dimension of the vectors (x_i)

	Mat from_func1 = calcCompLogL(model, features);	// Contains a 2 x # samples -Mat
	Mat from_func2 = calcMixtureLogL( model, features); // Contains a 1 x # samples -Mat

	Mat posterior = Mat::zeros(C_current,Number_features,CV_32FC1);


for(int j = 0; j < Number_features; j++){
		for(int i = 0; i < C_current; i++){

			posterior.at<float>(i,j) = exp(log(model.at(i)->weight) + from_func1.at<float>(i,j)-from_func2.at<float>(0,j));

		}
	}
	return posterior;
}

// Updates a given model on the basis of posteriors previously computed in the E-Step.
/*
model:     structure containing model parameters, will be updated in-place
           new model structure in which all parameters have been updated to reflect the current posterior distributions.
features:  matrix of feature vectors
posterior: the posterior p(y_i=j|x_i)
*/
void Aia5::gmmMStep(vector<struct comp*>& model, const Mat& features, const Mat& posterior){


	int C_current = model.size();	// Number of current clusters
	int Number_features = features.cols;	// Number of features/samples (x)
	int Number_dimensions = features.rows;	// Number of dimension of the vectors (x_i)

	Mat Onesmat = Mat::ones(Number_features,1,CV_32FC1);

	for(int i = 0; i < C_current; i++){

		Mat N_j = posterior.row(i)*Onesmat; // OK
		//cout << "N_j " << N_j << endl;

		float alpha_j = N_j.at<float>(0,0)/Number_features; // OK
		//cout << "alpha_j " << alpha_j << endl;

		Mat mu_j = 1/N_j.at<float>(0,0)*(features*posterior.row(i).t()); //OK
		//cout << "mu_j " << mu_j << endl;

		Mat COV = Mat::zeros(Number_dimensions,Number_dimensions,CV_32FC1);

		for(int j = 0; j < Number_features; j++){
			Mat diff_vec = features.col(j)-mu_j;
			COV += diff_vec*diff_vec.t()*posterior.at<float>(i,j); // OK
		}
		model.at(i)->weight = alpha_j;
		model.at(i)->mean = mu_j;
		model.at(i)->covar = 1/N_j.at<float>(0,0)*COV;

	}

}

/* *****************************
  GIVEN FUNCTIONS
***************************** */

// sets parameters and generates data for EM clustering
void Aia5::run(void){

	// *********************************

	// To Do: Adjust number of princ. comp. and number of clusters and evaluate performance
	// dimensionality of the generated feature vectors (number of principal components)
    int vectLen = 3;
    // number of components (clusters) to be used by the model
    int numberOfComponents = 3;

    // *********************************

     // read training data
    cout << "Reading training data..." << endl;
    vector<Mat> trainImgDB;
		readImageDatabase("./img/train/in/", trainImgDB);
    cout << "Done\n" << endl;

    // generate PCA-basis
    cout << "Generate PCA-basis from data:" << endl;
    vector<PCA> featProjection;
    genFeatureProjection(trainImgDB, featProjection, vectLen);
    cout << "Done\n" << endl;

    // this is going to contain the individual models (one per category)
    vector< vector<struct comp*> > models;

    // start learning
    cout << "Start learning..." << endl;
    for(int c=0; c<10; c++){
	cout << "\nTrain GMM of category " << c << endl;
	cout << " > Project on principal components of category " << c << " :" << endl;
	Mat fea = Mat(trainImgDB.at(c).rows, vectLen, CV_32FC1);
	featProjection.at(c).project( trainImgDB.at(c), fea );
	fea = fea.t();
	cout << "> Done" << endl;

	cout << " > Estimate probability density of category " << c << " :" << endl;
	// train the corresponding mixture model using EM...
	vector<struct comp*> model; // STOPS HERE
	trainGMM(fea, numberOfComponents, model);
	models.push_back(model);
	cout << "> Done" << endl;
    }
    cout << "Done\n" << endl;

    // read testing data
    cout << "Reading test data...:\t";
    vector<Mat> testImgDB;
		readImageDatabase("./img/test/in/" , testImgDB);
    cout << "Done\n" << endl;

    cout << "Test GMM: Start" << endl;
    Mat confMatrix = Mat::zeros(10, 10, CV_32FC1);
    int dtr = 0, n=0;
    // for each category within the test data
    for(int c_true=0; c_true<10; c_true++){
	n += testImgDB.at(c_true).rows;
	// init likelihood
	Mat maxMixLogL = Mat(1, testImgDB.at(c_true).rows, CV_32FC1);
	// estimated class
	Mat est = Mat::zeros(1, testImgDB.at(c_true).rows, CV_8UC1);

	for(int c_est=0; c_est<10; c_est++){
	    cout << " > Project on principal components of category " << c_est << " :\t";
	    Mat fea = Mat(testImgDB.at(c_true).rows, vectLen, CV_32FC1);
	    featProjection.at(c_est).project( testImgDB.at(c_true), fea );
	    fea = fea.t();
	    cout << "Done" << endl;

	    cout << " > Estimate class likelihood of category " << c_est << " :" << endl;

	    // get data log
			cout << "stops here" << endl;
	    Mat mixLogL = calcMixtureLogL(models.at(c_est), fea);
			cout << "stops here" << endl;

	    // compare to current max
	    for(int i=0; i<fea.cols; i++){
		if ( ( maxMixLogL.at<float>(0,i) < mixLogL.at<float>(0,i) ) or (c_est == 0) ){
		    maxMixLogL.at<float>(0,i) = mixLogL.at<float>(0,i);
		    est.at<uchar>(0,i) = c_est;
		}
	    }
	    cout << "Done\n" << endl;
	}
	// make corresponding entry in confusion matrix
	for(int i=0; i<testImgDB.at(c_true).rows; i++){
	    //cout << (int)est.at<uchar>(0,i) << "\t" << c_true << endl;
	    confMatrix.at<float>( c_true, (int)est.at<uchar>(0,i) )++;
	    dtr += (int)est.at<uchar>(0,i) == c_true;
	}
    }
    cout << "Test GMM: Done\n" << endl;

    cout << endl << "Confusion matrix:" << endl;
    cout << confMatrix << endl;
    cout << endl << "No of correctly classified:\t" << dtr << " of " << n << "\t( " << (dtr/(double)n*100) << "% )" << endl;

}

// sets parameters and generates data for EM clustering
void Aia5::test(void){

	// dimensionality of the generated feature vectors
    int vectLen = 2;

    // number of components (clusters) to generate
    //int actualComponentNum = 10;
		int actualComponentNum = 4;

    // maximal standard deviation of vectors in each cluster
    double actualDevMax = 0.3;

    // number of vectors in each cluster
    int trainingSize = 150;

    // initialise random component parameters (mean and standard deviation)
    Mat actualMean(actualComponentNum, vectLen, CV_32FC1);
    Mat actualSDev(actualComponentNum, vectLen, CV_32FC1);
    randu(actualMean, 0, 1);
    randu(actualSDev, 0, actualDevMax);

    // print distribution parameters to screen
		/*
    cout << "true mean" << endl;
    cout << actualMean << endl;
    cout << "true sdev" << endl;
    cout << actualSDev << endl;
		*/

    // initialise random cluster vectors
    Mat trainingData = Mat::zeros(vectLen, trainingSize*actualComponentNum, CV_32FC1);
    int n=0;
    RNG rng;
    for(int c=0; c<actualComponentNum; c++){
		for(int s=0; s<trainingSize; s++){
			for(int d=0; d<vectLen; d++){
				trainingData.at<float>(d,n) = rng.gaussian( actualSDev.at<float>(c, d) ) + actualMean.at<float>(c, d);
			}
			n++;
		}
    }

    // train the corresponding mixture model using EM...
 	 vector<struct comp*> model;
    trainGMM(trainingData, actualComponentNum, model);

}

// Trains a Gaussian mixture model with a specified number of components on the basis of a given set of feature vectors.
/*
data:     		feature vectors, one vector per column
numberOfComponents:	the desired number of components in the trained model
model:			the model, that will be created and trained inside of this function
*/
void Aia5::trainGMM(const Mat& data, int numberOfComponents, vector<struct comp*>& model){

    // the number and dimensionality of feature vectors
    int featureNum = data.cols;
    int featureDim = data.rows;

    // initialize the model with one component and arbitrary parameters
    struct comp* fst = new struct comp();
    fst->weight = 1;
    fst->mean = Mat::zeros(featureDim, 1, CV_32FC1);
    fst->covar = Mat::eye(featureDim, featureDim, CV_32FC1);
    model.push_back(fst);

    // the data-log-likelihood
    double dataLogL[2] = {0,0};

    // iteratively add components to the mixture model
    for(int i=1; i<=numberOfComponents; i++){

		cout << "Current number of components: " << i << endl;

		// the current combined data log-likelihood p(X|Omega)
		Mat mixLogL = calcMixtureLogL(model, data);
		dataLogL[0] = sum(mixLogL).val[0];
		dataLogL[1] = 0.;

		// EM iteration while p(X|Omega) increases
		int it = 0;

		while( (dataLogL[0] > dataLogL[1]) or (it == 0) ){

			printf("Iteration: %d\t:\t%f\r", it++, dataLogL[0]);

			// E-Step (computes posterior)
			Mat posterior = gmmEStep(model, data);

			// M-Step (updates model parameters)
			gmmMStep(model, data, posterior); // THIS STOPSSSS in numb of components 2

			// update the current p(X|Omega)
			dataLogL[1] = dataLogL[0];

			mixLogL = calcMixtureLogL(model, data);

			dataLogL[0] = sum(mixLogL).val[0];

		}


		cout << endl;

		// visualize the current model (with i components trained)
		if (featureDim >= 2){
		   plotGMM(model, data);
		}

		// add a new component if necessary
		if (i < numberOfComponents){
			initNewComponent(model, data);
		}
    }

    cout << endl << "**********************************" << endl;
    cout << "Trained model: " << endl;
    for(int i=0; i<model.size(); i++){
		cout << "Component " << i << endl;
		cout << "\t>> weight: " << model.at(i)->weight << endl;
		cout << "\t>> mean: " << model.at(i)->mean << endl;
		cout << "\t>> std: [" << sqrt(model.at(i)->covar.at<float>(0,0)) << ", " << sqrt(model.at(i)->covar.at<float>(1,1)) << "]" << endl;
		cout << "\t>> covar: " << endl;
		cout << model.at(i)->covar << endl;
		cout << endl;
    }

}

// Adds a new component to the input mixture model by spliting one of the existing components in two parts.
/*
model:		Gaussian Mixture Model parameters, will be updated in-place
features:	feature vectors
*/
void Aia5::initNewComponent(vector<struct comp*>& model, const Mat& features){

    // number of components in current model
    int compNum = model.size();

    // number of features
    int featureNum = features.cols;

    // dimensionality of feature vectors (equals 3 in this exercise)
    int featureDim = features.rows;

    // the largest component is split (this is not a good criterion...)
    int splitComp = 0;
    for(int i=0; i<compNum; i++){
		if (model.at(splitComp)->weight < model.at(i)->weight){
			splitComp = i;
		}
    }

    // split component 'splitComp' along its major axis
    Mat eVec, eVal;
    eigen(model.at(splitComp)->covar, eVal, eVec);

    Mat devVec = 0.5 * sqrt( eVal.at<float>(0) ) * eVec.row(0).t();

    // create new model structure and compute new mean values, covariances, new component weights...
    struct comp* newModel = new struct comp;
    newModel->weight = 0.5 * model.at(splitComp)->weight;
    newModel->mean = model.at(splitComp)->mean - devVec;
    newModel->covar = 0.25 * model.at(splitComp)->covar;

    // modify the split component
    model.at(splitComp)->weight = 0.5*model.at(splitComp)->weight;
    model.at(splitComp)->mean += devVec;
    model.at(splitComp)->covar *= 0.25;

    // add it to old model
    model.push_back(newModel);

}

// Visualises the contents of a feature space and the associated mixture model.
/*
model: 		parameters of a Gaussian mixture model
features: 	feature vectors

Feature vectors are plotted as black points
Estimated means of components are indicated by blue circles
Estimated covariances are indicated by blue ellipses
If the feature space has more than 2 dimensions, only the first two dimensions are visualized.
*/
void Aia5::plotGMM(const vector<struct comp*>& model, const Mat& features){

    // size of the plot
    int imSize = 500;

    // get scaling factor to scale coordinates
    double max_x=0, max_y=0, min_x=0, min_y=0;
    for(int n=0; n<features.cols; n++){
		if (max_x < features.at<float>(0, n) )
			max_x = features.at<float>(0, n);
		if (min_x > features.at<float>(0, n) )
			min_x = features.at<float>(0, n);
		if (max_y < features.at<float>(1, n) )
			max_y = features.at<float>(1, n);
		if (min_y > features.at<float>(1, n) )
			min_y = features.at<float>(1, n);
    }
    double scale = (imSize-1)/max((max_x - min_x), (max_y - min_y));
    // create plot
    Mat plot = Mat(imSize, imSize, CV_8UC3, Scalar(255,255,255) );

    // set feature points
    for(int n=0; n<features.cols; n++){
		plot.at<Vec3b>( ( features.at<float>(0, n) - min_x ) * scale, max( (features.at<float>(1,n)-min_y)*scale, 5.) ) = Vec3b(0,0,0);
    }
    // get ellipse of components
    Mat EVec, EVal;
    for(int i=0; i<model.size(); i++){

		eigen(model.at(i)->covar, EVal, EVec);
		double rAng = atan2(EVec.at<float>(0, 1), EVec.at<float>(0, 0));

		// draw components
		circle(plot,  Point( (model.at(i)->mean.at<float>(1,0)-min_y)*scale, (model.at(i)->mean.at<float>(0,0)-min_x)*scale ), 3, Scalar(255,0,0), 2);
		ellipse(plot, Point( (model.at(i)->mean.at<float>(1,0)-min_y)*scale, (model.at(i)->mean.at<float>(0,0)-min_x)*scale ), Size(sqrt(EVal.at<float>(1))*scale*2, sqrt(EVal.at<float>(0))*scale*2), rAng*180/CV_PI, 0, 360, Scalar(255,0,0), 2);

    }

    // show plot an abd wait for key
    //imshow("Current model", plot);
    //waitKey(0);

}

// constructs PCA-space based on all samples of each class
/*
imgDB		image data base; each matrix corresponds to one class; each row to one image
featSpace	the PCA-bases for each class
vectLen		number of principal components to be used
*/
void Aia5::genFeatureProjection(const vector<Mat>& imgDB, vector<PCA>& featSpace, int vectLen){

    int c = 0;
    for(vector<Mat>::const_iterator cat = imgDB.begin(); cat != imgDB.end(); cat++, c++){
	cout << " > Generate PC of category " << c << " :\t";
      	PCA compPCA = PCA(*cat, Mat(), CV_PCA_DATA_AS_ROW, vectLen);
	featSpace.push_back(compPCA);
	cout << "Done" << endl;
    }
}

// reads image data base from disc
/*
dataPath	path to directory
db		each matrix in this vector corresponds to one class, each row of the matrix corresponds to one image
*/
void Aia5::readImageDatabase(string dataPath, vector<Mat>& db){

    // directory delimiter. you might wanna use '\' on windows systems
    char delim = '/';

    char curDir[100];
    db.reserve(10);

    int numberOfImages = 0;
    for(int c=0; c<10; c++){

	list<Mat> imgList;
	sprintf(curDir, "%s%c%i%c", dataPath.c_str(), delim, c, delim);

	// read directory
	DIR* pDIR;
	struct dirent *entry;
	struct stat s;

	stat(curDir,&s);

	// if path is a directory
	if ( (s.st_mode& S_IFMT ) == S_IFDIR ){

	    if( (pDIR = opendir(curDir)) ){		// Changed ()
		// for all entries in directory
		while((entry = readdir(pDIR))){ 																								// Changed to ()
		    // is current entry a file?
		    stat((curDir + string("/") + string(entry->d_name)).c_str(),&s);
		    if ( ( (s.st_mode & S_IFMT ) != S_IFDIR ) and ( (s.st_mode & S_IFMT ) == S_IFREG ) ){
			// if all conditions are fulfilled: load data
			Mat img = imread((curDir + string(entry->d_name)).c_str(), 0);
			img.convertTo(img, CV_32FC3);
			img /= 255.;
			imgList.push_back(img);
			numberOfImages++;
		    }
		}
		closedir(pDIR);
	    }else{
		cerr << "\nERROR: cant open data dir " << dataPath << endl;
		exit(-1);
	    }
	}else{
	    cerr << "\nERROR: provided path does not specify a directory: ( " << dataPath << " )" << endl;
	    exit(-1);
	}

	int numberOfImages = imgList.size();
	int numberOfPixPerImg = imgList.front().cols * imgList.front().rows;

	Mat feature = Mat(numberOfImages, numberOfPixPerImg, CV_32FC1);

	int i = 0;
	for(list<Mat>::iterator img = imgList.begin(); img != imgList.end(); img++, i++){
	    for(int p = 0; p<numberOfPixPerImg; p++){
		feature.at<float>(i, p) = img->at<float>(p);
	    }
	}
	db.push_back(feature);
    }
}

