

#include "utils.h"

double threshold(double num)
{
  return num > 0 ? num : 0;
}

// computes cumulative sum of vector x
VectorXd cumsum(const VectorXd& x) {
  const int n(x.size());
  VectorXd cmsm(n);
  //cmsm = std::partial_sum(x.data(), x.data() + x.size(), cmsm.data(), std::plus<double>());
  cmsm(0) = x(0);

  for (int i = 1; i < n; i++) {
    cmsm(i) = cmsm(i-1) + x(i);
  }
  return (cmsm);
}

// computes reverse cumulative sum of vector x
VectorXd cumsumrev(const VectorXd& x) {
  const int n(x.size());
  VectorXd cmsm(n);
  //std::reverse(x.data(), x.data() + x.size());
  //cmsm = std::partial_sum(x.data(), x.data() + x.size(), cmsm.data(), std::plus<double>());
  cmsm(0) = x(n-1);
  //double tmpsum = 0;

  for (int i = 1; i < n; i++) {
    //tmpsum += cmsm(i-1);
    cmsm(i) = cmsm(i-1) + x(n-i-1);
  }
  std::reverse(cmsm.data(), cmsm.data() + cmsm.size());
  return (cmsm);
}




bool stopRule(const VectorXd& cur, const VectorXd& prev, const double& tolerance) {
  for (unsigned i = 0; i < cur.rows(); i++) {
    if ( (cur(i) != 0 && prev(i) == 0) || (cur(i) == 0 && prev(i) != 0) ) {
      return 0;
    }
    if (cur(i) != 0 && prev(i) != 0 &&
        std::abs( (cur(i) - prev(i)) / prev(i)) > tolerance) {
  	  return 0;
    }
  }
  return 1;
}
