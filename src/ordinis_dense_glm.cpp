#define EIGEN_DONT_PARALLELIZE

#include "CoordLogisticDense.h"
#include "CoordGLMDense.h"
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

List coord_ordinis_dense_glm(Rcpp::NumericMatrix x_,
                             Rcpp::NumericVector y_,
                             Rcpp::NumericVector weights_,
                             Rcpp::NumericVector offset_,
                             Rcpp::NumericVector lambda_,
                             Rcpp::NumericVector penalty_factor_,
                             Rcpp::NumericMatrix limits_,
                             int    nlambda_,
                             double lmin_ratio_,
                             bool   standardize_,
                             bool   intercept_,
                             bool   glm_fam_,
                             List   opts_)
{

    const int n = x_.rows();
    const int p = x_.cols();

    MatrixXd datX(n, p);
    VectorXd datY(n);
    VectorXd weights(n);
    VectorXd offset(n);
    MatrixXd limits(2, p);

    // Copy data and convert type from double to float
    std::copy(x_.begin(), x_.end(), datX.data());
    std::copy(y_.begin(), y_.end(), datY.data());
    std::copy(weights_.begin(), weights_.end(), weights.data());
    std::copy(offset_.begin(), offset_.end(), offset.data());

    std::copy(limits_.begin(), limits_.end(), limits.data());

    //Map<VectorXd>  weights(as<Map<VectorXd> >(weights_));

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
    const double alpha     = as<double>(opts["alpha"]);
    const double gamma     = as<double>(opts["gamma"]);
    const int maxit_irls   = as<int>(opts["maxit.irls"]);
    const double tol_irls  = as<double>(opts["tol.irls"]);
    const int dfmax        = as<int>(opts["dfmax"]);
    const bool standardize = standardize_;
    const bool intercept   = intercept_;
    const bool glm_fam     = glm_fam_;

    std::vector<std::string> penalty(as< std::vector<std::string> >(opts["penalty"]));

    DataStd<double> datstd(n, p, standardize, intercept, true);
    datstd.standardize(datX, datY, limits);


    CoordGLMDense *solver;
    CoordLogisticDense *solver_bin;

    //CoordBase<Eigen::SparseVector<double> > *solver = NULL; // obj doesn't point to anything yet

    if (glm_fam)
    {
        Rcpp::Function var        = as<Rcpp::Function>(opts["variance"]);
        Rcpp::Function mu_eta     = as<Rcpp::Function>(opts["mu_eta"]);
        Rcpp::Function linkinv    = as<Rcpp::Function>(opts["linkinv"]);
        Rcpp::Function dev_resids = as<Rcpp::Function>(opts["dev_resids"]);

        solver = new CoordGLMDense(datX, datY,
                                   weights, offset,
                                   penalty_factor,
                                   limits, penalty[0],
                                   var, mu_eta, linkinv, dev_resids,
                                   intercept, alpha,
                                   tol, maxit_irls, tol_irls);
    } else
    {
        solver_bin = new CoordLogisticDense(datX, datY,
                                            weights, offset,
                                            penalty_factor,
                                            limits, penalty[0],
                                            intercept, alpha,
                                            tol, maxit_irls, tol_irls);
    }



    if (nlambda < 1)
    {
        double lmax = 0.0;
        if (glm_fam)
        {
            lmax = solver->get_lambda_zero();
        } else
        {
            lmax = solver_bin->get_lambda_zero();
        }

        double lmin = lmin_ratio_ * lmax;
        lambda.setLinSpaced(nlambda_, std::log(lmax), std::log(lmin));
        lambda = lambda.exp();
        nlambda = lambda.size();
    }




    SpMat beta(p + 1, nlambda);
    beta.reserve(Eigen::VectorXi::Constant(nlambda, std::min(n, p)));

    //MatrixXd beta(p + 1, nlambda);
    //VectorXd lossvec(nlambda);

    IntegerVector niter(nlambda);
    double ilambda = 0.0;

    double null_dev = 0.0;
    VectorXd deviance(nlambda);
    deviance.setZero();


    int last = nlambda;
    for(int i = 0; i < nlambda; i++)
    {

        ilambda = lambda[i];

        if (glm_fam)
        {
            if(i == 0)
                solver->init(ilambda, gamma);
            else
                solver->init_warm(ilambda, gamma);

            niter[i]    = solver->solve(maxit);

            SpVec res   = solver->get_beta();
            int nzero   = solver->get_nzero();
            deviance(i) = solver->get_dev();

            if (i == 0) null_dev = solver->get_null_dev();

            if (p >= n)
            {
                if ((nzero > dfmax || deviance(i) < 0.1 * null_dev) && i > 0 )
                {
                    last = i - 1;
                    break;
                }
            } else
            {
                if ((nzero > dfmax || deviance(i) < 0.05 * null_dev) && i > 0 )
                {
                    last = i - 1;
                    break;
                }
            }

            double beta0 = 0.0;
            beta0 = solver->get_intercept();

            datstd.recover(beta0, res);
            //beta(0,i) = beta0;
            //beta.block(1, i, p, 1) = res;
            write_beta_matrix(beta, i, beta0, res);
        } else
        {
            if(i == 0)
                solver_bin->init(ilambda, gamma);
            else
                solver_bin->init_warm(ilambda, gamma);

            niter[i]    = solver_bin->solve(maxit);

            SpVec res   = solver_bin->get_beta();
            int nzero   = solver_bin->get_nzero();
            deviance(i) = solver_bin->get_dev();

            if (i == 0) null_dev = solver_bin->get_null_dev();

            if (p >= n)
            {
                if ((nzero > dfmax || deviance(i) < 0.1 * null_dev) && i > 0 )
                {
                    last = i - 1;
                    break;
                }
            } else
            {
                if ((nzero > dfmax || deviance(i) < 0.05 * null_dev) && i > 0 )
                {
                    last = i - 1;
                    break;
                }
            }

            double beta0 = 0.0;
            beta0 = solver_bin->get_intercept();

            datstd.recover(beta0, res);
            //beta(0,i) = beta0;
            //beta.block(1, i, p, 1) = res;
            write_beta_matrix(beta, i, beta0, res);
        }

        //lossvec(i) = solver->get_loss();
    }

    if (glm_fam)
    {
        delete solver;
    } else
    {
        delete solver_bin;
    }


    beta.makeCompressed();

    return List::create(Named("beta")          = beta,
                        Named("niter")         = niter,
                        Named("lambda")        = lambda,
                        //Named("loss")          = lossvec,
                        Named("deviance")      = deviance,
                        Named("null.deviance") = null_dev,
                        Named("last")          = last);
}

// [[Rcpp::export]]
List coord_ordinis_dense_glm_cpp(Rcpp::NumericMatrix x,
                                 Rcpp::NumericVector y,
                                 Rcpp::NumericVector weights,
                                 Rcpp::NumericVector offset,
                                 Rcpp::NumericVector lambda,
                                 Rcpp::NumericVector penalty_factor,
                                 Rcpp::NumericMatrix limits,
                                 int nlambda,
                                 double lmin_ratio,
                                 bool standardize,
                                 bool intercept,
                                 bool glm_fam,
                                 List opts)
{
    return coord_ordinis_dense_glm(x, y, weights, offset,
                                   lambda, penalty_factor,
                                   limits,
                                   nlambda,
                                   lmin_ratio,
                                   standardize,
                                   intercept,
                                   glm_fam,
                                   opts);
}
