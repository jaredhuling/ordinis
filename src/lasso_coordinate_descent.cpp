#define EIGEN_DONT_PARALLELIZE

#include "CoordLasso.h"
#include "DataStd.h"


using Eigen::MatrixXf;
using Eigen::VectorXf;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::ArrayXf;
using Eigen::ArrayXd;
using Eigen::ArrayXXf;
using Eigen::Map;

using Rcpp::wrap;
using Rcpp::as;
using Rcpp::List;
using Rcpp::Named;
using Rcpp::IntegerVector;

typedef Map<VectorXd> MapVecd;
typedef Map<Eigen::MatrixXd> MapMatd;
typedef Eigen::SparseVector<double> SpVec;
typedef Eigen::SparseMatrix<double> SpMat;

inline void write_beta_matrix(SpMat &betas, int col, double beta0, SpVec &coef)
{
    betas.insert(0, col) = beta0;

    for(SpVec::InnerIterator iter(coef); iter; ++iter)
    {
        betas.insert(iter.index() + 1, col) = iter.value();
    }
}

List coord_lasso(Rcpp::NumericMatrix x_,
                 Rcpp::NumericVector y_,
                 Rcpp::NumericVector weights_,
                 Rcpp::NumericVector lambda_,
                 Rcpp::NumericVector penalty_factor_,
                 int    nlambda_,
                 double lmin_ratio_,
                 bool   standardize_,
                 bool   intercept_,
                 List   opts_)
{

    const int n = x_.rows();
    const int p = x_.cols();

    MatrixXd datX(n, p);
    VectorXd datY(n);

    // Copy data and convert type from double to float
    std::copy(x_.begin(), x_.end(), datX.data());
    std::copy(y_.begin(), y_.end(), datY.data());

    const Map<VectorXd>  weights(as<Map<VectorXd> >(weights_));

    // In glmnet, we minimize
    //   1/(2n) * ||y - X * beta||^2 + lambda * ||beta||_1
    // which is equivalent to minimizing
    //   1/2 * ||y - X * beta||^2 + n * lambda * ||beta||_1
    ArrayXd lambda(as<ArrayXd>(lambda_));
    int nlambda = lambda.size();

    ArrayXd penalty_factor(as<ArrayXd>(penalty_factor_));


    List opts(opts_);
    const int maxit        = as<int>(opts["maxit"]);
    const double tol       = as<double>(opts["tol"]);
    const bool standardize = standardize_;
    const bool intercept   = intercept_;

    DataStd<double> datstd(n, p, standardize, intercept);
    datstd.standardize(datX, datY);

    CoordLasso *solver;
    solver = new CoordLasso(datX, datY, penalty_factor, tol);


    if (nlambda < 1)
    {
        double lmax = 0.0;
        lmax = solver->get_lambda_zero() / n * datstd.get_scaleY();

        double lmin = lmin_ratio_ * lmax;
        lambda.setLinSpaced(nlambda_, std::log(lmax), std::log(lmin));
        lambda = lambda.exp();
        nlambda = lambda.size();
    }




    //SpMat beta(p + 1, nlambda);
    //beta.reserve(Eigen::VectorXi::Constant(nlambda, std::min(n, p)));

    MatrixXd beta(p+1, nlambda);
    VectorXd lossvec(nlambda);

    IntegerVector niter(nlambda);
    double ilambda = 0.0;

    int last = nlambda;
    for(int i = 0; i < nlambda; i++)
    {
        ilambda = lambda[i] * n / datstd.get_scaleY();

        if(i == 0)
            solver->init(ilambda);
        else
            solver->init_warm(ilambda);

        niter[i] = solver->solve(maxit);
        VectorXd res = solver->get_beta();
        int nzero = solver->get_nzero();
        double beta0 = 0.0;
        datstd.recover(beta0, res);
        beta(0,i) = beta0;
        beta.block(1, i, p, 1) = res;
        //write_beta_matrix(beta, i, beta0, res);

        lossvec(i) = solver->get_loss();

        if (nzero > n && i > 0)
        {
            last = i;
            break;
        }
    }

    delete solver;

    //beta.makeCompressed();

    return List::create(Named("beta")        = beta,
                        Named("niter")       = niter,
                        Named("lambda")      = lambda,
                        Named("loss")        = lossvec,
                        Named("last")        = last);
}

// [[Rcpp::export]]
List coord_lasso_cpp(Rcpp::NumericMatrix x,
                     Rcpp::NumericVector y,
                     Rcpp::NumericVector weights,
                     Rcpp::NumericVector lambda,
                     Rcpp::NumericVector penalty_factor,
                     int nlambda,
                     double lmin_ratio,
                     bool standardize,
                     bool intercept,
                     List opts)
{
    return coord_lasso(x, y, weights, lambda, penalty_factor,
                       nlambda,
                       lmin_ratio,
                       standardize,
                       intercept,
                       opts);
}
