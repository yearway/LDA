//
//  main.cpp
//  lda
//
//  Created by WEI YE on 10/7/13.
//  Copyright (c) 2013 WEI YE. All rights reserved.
//

#include <iostream>
#include <fstream>
#include <string>
#include <opencv2/core/core.hpp>
#include "decomposition.hpp"

using namespace std;
using namespace cv;

bool readData(string filename, Mat_<float>& data)
{
    ifstream dfile;
    dfile.open(filename.c_str(), ios::binary);
    if (!dfile.is_open()) 
        return false;

    int rows, cols, i, j;
    dfile.read((char*)&rows, sizeof(int));
    dfile.read((char*)&cols, sizeof(int));

    if (rows <= 0 || cols <= 0) {
        return false;
    }

    data.create(rows, cols);
    for (i = 0; i < rows; i++) {
        for (j = 0; j < cols; j++) {
            dfile.read((char*)(data.ptr<float>(i)+j), sizeof(float));
        }
    }
    return true;
}

// -- Input  --
// x  -  features(N x D)
// y  -  labels  (N x 1)
// w  -  weights (N x 1)
//
// -- Output --
// W  -  transformation matrix;
// Mi -  means of each class
void fisher_lda(const Mat_<float>& x, const Mat_<int>& y, const Mat_<float>& w, Mat_<float>& W, vector<Mat_<float>>& Mi)
{
    int n_point = x.rows;
    int n_dim = x.cols;
    int n_class = *max_element(y.begin(), y.end()) + 1;
    if (n_dim < 1|| n_class < 2 || n_point <= 1 || n_point != y.rows || n_point != w.rows){
        string error_msg = "Invalid training set!";
        CV_Error(CV_StsBadArg, error_msg);
    }

    vector<float> numPoints(n_class, 0);
    Mat_<float> M = Mat_<float>::zeros(1, n_dim); // mean of all points
    Mi = vector<Mat_<float>>(n_class);            // means of each class

    // compute means and covariances
    int p, c;
    for (c = 0; c < n_class; c++)
        Mi[c] = Mat_<float>::zeros(1, n_dim);
    
    for (p = 0; p < n_point; p++) {
        c = y(p,0);
        numPoints[c] += w(p,0);
        Mi[c] += (x.row(p)*w(p,0));
        M += (x.row(p)*w(p,0));
    }

    for (c = 0; c < n_class; c++)
        Mi[c] /= numPoints[c];
    M /= n_point;
    
    // compute within-class scatter matrix Sw = sum{ Si }
    Mat_<float> subMean(n_point, n_dim);
    for (p = 0; p < n_point; p++) {
        c = y(p,0);
        subMean.row(p) = x.row(p) - Mi[c];
    }
    Mat_<float> Sw = subMean.t()*subMean;

    // compute between-class scatter matrix Sb
    Mat_<float> Sb(Mat_<float>::zeros(n_dim, n_dim));
    for (c = 0; c < n_class; c++) {
        Mat_<float> diff = Mi[c] - M;
        Sb += numPoints[c]*(diff.t()*diff);
    }

    // compute eigen values/vectors for Sw^-1*Sb
    Mat_<float> eigenValues;
    if (abs(determinant(Sw)) > 1e-3) {
        // eigen(Sw.inv()*Sb, eigenValues, W); // Wrong! OpenCV's "eigen" function only apply for symmetric matrix
        EigenvalueDecomposition eig(Sw.inv()*Sb);
        Mat eigVal = eig.eigenvalues();
        Mat eigVec = eig.eigenvectors();
        eigVal = eigVal.reshape(1, 1);
        Mat order_des;
        sortIdx(eigVal, order_des, CV_SORT_EVERY_ROW + CV_SORT_DESCENDING);
        W = Mat_<float>(n_dim, n_dim);
        for (int i = 0; i < n_dim; i++) {
            eigVec.col(order_des.at<int>(0,i)).copyTo(W.col(i));
        }
    }else{ // Sw is a singular matrix
        string err_msg = "Sw is invertible!";
        CV_Error(CV_StsError, err_msg);
    }
}



// -- Input  --
// X    -  single test data point
// Mi_p -  projected means of each class
//
// -- Return --
// label of X

int predict(const Mat_<float>& X, const vector<Mat_<float>>& Mi_p)
{
    double min_dist = norm(X - Mi_p[0]);
    int label = 0;
    for (int c = 1; c < Mi_p.size(); c++) {
        double dist = norm(X - Mi_p[c]);
        if (dist < min_dist) {
            min_dist = dist;
            label = c;
        }
    }
    return label;
}

int main()
{
    Mat_<float> data, W;
    vector<Mat_<float>> Mi;
    if (!readData("/Users/weiye/Documents/YEWEI/practice/LDA/0_CEHOG1-B16.featbin", data))
        return 1;
//    data = (Mat_<float>(11,2) << 1,2,2,3,3,3,4,5,5,5,1,0,2,1,3,1,3,2,5,3,6,5);

    Mat_<int> labels(data.rows, 1);
    labels.rowRange(0, 1816).setTo(0);
    labels.rowRange(1816, data.rows).setTo(1);
    
    Mat_<float> weights(Mat_<float>::ones(data.rows, 1));
    
    fisher_lda(data, labels, weights, W, Mi);
    
    vector<Mat_<float>> Mi_p(2);
    int N = 1;
    Mat_<float> w = W.colRange(0, N);
    for (int c = 0; c < Mi.size(); c++) 
        Mi_p[c] = Mi[c]*w;

    double n_correct = 0.0;
    for (int p = 0; p < data.rows; p++)
        if (predict(data.row(p)*w, Mi_p) == labels(p,0))
            n_correct++;

    cout<<Mi_p[0]<<Mi_p[1]<<endl;
    double accuracy = n_correct/data.rows;
    cout<<"Accuracy = "<<accuracy*100.0<<"%"<<endl;
    return 0;
}
