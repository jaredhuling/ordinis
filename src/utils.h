#ifndef _ordinis_UTILS_H
#define _ordinis_UTILS_H


#include <Rcpp.h>
#include <RcppEigen.h>
#include <vector>
#include <functional>
#include <algorithm>
#include <iostream>
#include <cmath>
#include <numeric>


using Eigen::MatrixXd;
using Eigen::ArrayXd;
using Eigen::VectorXd;
using Eigen::VectorXi;

typedef Eigen::SparseVector<double> SparseVector;

double threshold(double num);

VectorXd cumsum(const VectorXd& x);

VectorXd cumsumrev(const VectorXd& x);


bool stopRule(const VectorXd& cur, const VectorXd& prev, const double& tolerance);

bool stopRule(const SparseVector& cur, const SparseVector& prev, const double& tolerance);

bool stopRule(SparseVector& cur, SparseVector& prev, const double& tolerance);



#endif
