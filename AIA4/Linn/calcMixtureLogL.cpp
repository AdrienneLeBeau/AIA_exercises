Mat Aia5::calcMixtureLogL(const vector<struct comp*>& model, const Mat& features){

	int C_current = model.size();	// Number of current clusters
	int Number_features = features.cols;	// Number of features/samples (x)
	int Number_dimensions = features.rows;	// Number of dimension of the vectors (x_i)

	// Getting the log-probabilities from the calcCompLogL function
  Mat log_Prob_all_Cs = calcCompLogL( model, features);

	// Creating the Mat that will contain the log-likelihood of each feature by
	// combining the likelihoods in all model components
	Mat w_vec = Mat::zeros(1,C_current,CV_32F);

	for(int i = 0; i < C_current; i++){
		w_vec.at<float>(0,i) = model.at(i)->weight;
	}

	double min, scale;
	minMaxLoc(log_Prob_all_Cs, &min, &scale);

	Mat exp_Mat;
	exp(log_Prob_all_Cs-scale,exp_Mat);
	Mat log_Mat;
	log(w_vec*exp_Mat,log_Mat);
	Mat log_sum = scale + log_Mat;

	return log_sum;
}
