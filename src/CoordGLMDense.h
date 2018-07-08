#ifndef COORDGLMDENSE_H
#define COORDGLmDENSE_H

#include "CoordBase.h"
#include "utils.h"

// minimize  1/2 * ||y - X * beta||^2 + lambda * ||beta||_1
//
// In ADMM form,
//   minimize f(x) + g(z)
//   s.t. x - z = 0
//
// x => beta
// z => -X * beta
// A => X
// b => y
// f(x) => 1/2 * ||Ax - b||^2
// g(z) => lambda * ||z||_1
class CoordGLMDense: public CoordBase<Eigen::SparseVector<double> > //Eigen::SparseVector<double>
{
protected:
    typedef float Scalar;
    typedef double Double;
    typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> Matrix;
    typedef Eigen::Matrix<double, Eigen::Dynamic, 1> Vector;
    typedef Eigen::Map<const Matrix> MapMat;
    typedef Eigen::Map<const Vector> MapVec;
    typedef const Eigen::Ref<const Matrix> ConstGenericMatrix;
    typedef const Eigen::Ref<const Vector> ConstGenericVector;
    typedef Eigen::SparseMatrix<double> SpMat;
    typedef Eigen::SparseVector<double> SparseVector;
    typedef Eigen::SparseVector<int> SparseVectori;

    typedef SparseVector::InnerIterator InIterVec;
    typedef SparseVectori::InnerIterator InIterVeci;

    MapMat datX;                  // data matrix
    MapVec datY;                  // response vector
    MapVec weights;               // weight vector
    MapVec offset;                // offset vector

    Scalar lambda, lambda_ridge, gamma;  // L1 penalty

    double threshval;
    VectorXd resid_cur, xbeta_cur, mu, varmu, mu_eta_nv, W, weightssqrt, z;

    std::string penalty;
    ArrayXd penalty_factor;       // penalty multiplication factors

    Rcpp::Function var, mu_eta, linkinv, dev_resids;

    bool intercept;
    MapMat limits;
    double alpha;
    int maxit_irls;
    double tol_irls;
    int penalty_factor_size;

    VectorXd XY;                    // X'Y
    VectorXd Xsq;                 // colSums(X^2)

    Scalar lambda0;               // minimum lambda to make coefficients all zero

    double lprev;

    double beta0; // intercept

    double weights_sum, resids_sum;

    // pointer we will set to one of the thresholding functions
    typedef double (*thresh_func_ptr)(double &value, const double &penalty, const double &gamma, const double &l2, const double &denom);

    thresh_func_ptr thresh_func;

    /*
    static void soft_threshold(SparseVector &res, const Vector &vec, const double &penalty)
    {
        int v_size = vec.size();
        res.setZero();
        res.reserve(v_size);

        const double *ptr = vec.data();
        for(int i = 0; i < v_size; i++)
        {
            if(ptr[i] > penalty)
                res.insertBack(i) = ptr[i] - penalty;
            else if(ptr[i] < -penalty)
                res.insertBack(i) = ptr[i] + penalty;
        }
    }
    */

    bool converged_irls()
    {
        //return (stopRule(beta, beta_prev_irls, tol_irls));

        if (std::abs(deviance - deviance_prev) / (0.1 + std::abs(deviance)) < tol_irls)
        {
            return true;
        } else
        {
            return false;
        }

    }

    void initialize_params()
    {
        double ymean = (weights.array() * datY.array()).matrix().mean();

        if (intercept)
        {
            beta0 = std::log(ymean / (1.0 - ymean));
        } else
        {
            beta0 = 0.0;
        }

        xbeta_cur.array() = offset.array() + beta0;

        // calculate null deviance
        //null_dev = (-1.0 * datY.array() * log(ymean) - (1.0 - datY.array()) * std::log(1.0 - ymean)).sum();

        //wtdmu <- if (intercept)
        //    sum(weights * y)/sum(weights)
        //    else linkinv(offset)
        //        nulldev <- sum(dev.resids(y, wtdmu, weights))


        if (intercept)
        {
            double wtdmu;
            wtdmu = (weights.array() * datY.array() / weights.sum()).sum();
            Rcpp::NumericVector nulldev_all = dev_resids(datY, wtdmu, weights);
            null_dev = sum(nulldev_all);
        } else
        {
            Rcpp::NumericVector wtdmu = linkinv(offset);
            Rcpp::NumericVector nulldev_all = dev_resids(datY, wtdmu, weights);
            null_dev = sum(nulldev_all);
        }
    }

    virtual void update_mu_eta()
    {
        Rcpp::NumericVector mu_eta_ = mu_eta(xbeta_cur);

        std::copy(mu_eta_.begin(), mu_eta_.end(), mu_eta_nv.data());
    }

    virtual void update_var_mu()
    {
        Rcpp::NumericVector var_mu_nv = var(mu);

        std::copy(var_mu_nv.begin(), var_mu_nv.end(), varmu.data());
    }

    virtual void update_mu()
    {
        // mu <- linkinv(eta <- eta + offset)
        Rcpp::NumericVector mu_nv = linkinv(xbeta_cur);

        std::copy(mu_nv.begin(), mu_nv.end(), mu.data());
    }

    void update_quadratic_approx()
    {
        // calculate mean function
        //p = 1.0 / (1.0 + ((-1.0 * xbeta_cur.array()).exp()));

        //mu = linkinv(xbeta_cur);
        update_mu();

        //varmu = var(mu);
        update_var_mu();

        //mu_eta_nv = mu_eta(xbeta_cur);
        update_mu_eta();

        // construct weights and multiply by user-specified weights
        //W = weights.array() * p.array() * (1.0 - p.array());
        W = (weights.array() * mu_eta_nv.array().square() / varmu.array()).array().sqrt();

        // make sure no weights are too small
        for (int k = 0; k < nobs; ++k)
        {
            if (W(k) < 1e-5)
            {
                W(k) = 1e-5;
            }
        }

        /*
        update_var_mu();

        update_mu_eta();

        update_z();

        update_w();

        solve_wls();

        update_eta();

        update_mu();

        update_dev_resids();
         */

        // here we update the residuals and multiply by user-specified weights, which
        // will be multiplied by X. ie X'resid_cur = X'Wz, where z is the working response from IRLS
        resid_cur = weights.array() * (datY.array() - mu.array()); // + xbeta_cur.array() * W.array().sqrt();

        //Xsq = (W.array().sqrt().matrix().asDiagonal() * datX).array().square().colwise().sum();

        // we will check this in later iterations
        // and only update when Xsq(j) < 0
        // (it's always positive, so this way we can check and avoid re-calculating unnecessarily)
        Xsq.fill(-1.0);

        // this is needed for intercept updates
        weights_sum = W.sum();

        // update deviance
        Rcpp::NumericVector dev_resids_all = dev_resids(datY, mu, weights);
        deviance = sum(dev_resids_all);
    }

    void update_intercept()
    {

        if (intercept)
        {
            resids_sum = (resid_cur).sum();

            double beta0_delta = resids_sum / weights_sum;

            beta0             += beta0_delta;

            // update the (weighted) working residual!
            resid_cur.array() -= beta0_delta * W.array();

            // update the linear predictor!
            xbeta_cur.array() += beta0_delta;
        }
    }

    double compute_loss()
    {
        //double sum_squares = 0.5 * (resid_cur.array().square().sum()) / double(nobs);
        double sum_squares = 0.5 * (resid_cur.array().square().sum());
        double penalty_part = 0.0;

        /*
        if (penalty_factor_size < 1)
        {
            penalty_part = lambda * beta.array().abs().matrix().sum();
        } else
        {
            penalty_part = lambda * (beta.array() * penalty_factor.array()).abs().matrix().sum();
        }
         */

        return (sum_squares + penalty_part);
    }

    static double soft_threshold(double &value, const double &penalty, const double &gamma, const double &l2, const double &denom)
    {

        if (std::abs(value) <= penalty)
            return(0.0);
        else if (value > penalty)
            return( (value - penalty) / (denom + denom * l2) );
        else
            return( (value + penalty) / (denom + denom * l2) );

        /* // this ordering is slower for high-dimensional problems
        if(value > penalty)
            return( (value - penalty) / (denom + l2) );
        else if(value < -penalty)
            return( (value + penalty) / (denom + l2) );
        else
            return(0.0);
         */
    }

    static double scad_threshold(double &value, const double &penalty, const double &gamma, const double &l2, const double &denom)
    {
        double val_abs = std::abs(value);

        if (val_abs <= penalty)
            return(0.0);
        else if (val_abs <= penalty * (1.0 + l2) + penalty)
        {
            if(value > penalty)
                return((value - penalty) / (  denom + denom * l2 ));
            else
                return((value + penalty) / (  denom + denom * l2 ));
        } else if (val_abs <= gamma * penalty * (1.0 + l2))
        {
            if ((gamma - 1.0) * value > gamma * penalty)
                return( (value - gamma * penalty / (gamma - 1.0)) / (denom * ( 1.0 - 1.0 / (gamma - 1.0) + l2 )) );
            else
                return( (value + gamma * penalty / (gamma - 1.0)) / (denom * ( 1.0 - 1.0 / (gamma - 1.0) + l2 )) );
        } else
        {
            return(value / (denom + denom * l2));
        }

    }

    static double mcp_threshold(double &value, const double &penalty, const double &gamma, const double &l2, const double &denom)
    {
        double val_abs = std::abs(value);

        if (val_abs <= penalty)
            return(0.0);
        else if (val_abs <= gamma * penalty * (1.0 + l2))
        {
            if(value > penalty)
                return((value - penalty) / ( denom * (1.0 + l2 - 1.0 / gamma) ));
            else
                return((value + penalty) / ( denom * (1.0 + l2 - 1.0 / gamma) ));
        } else
            return(value / (denom + denom * l2));

        /*
        if (std::abs(value) > gamma * penalty * (1.0 + l2))
            return(value / (denom + denom * l2));
        else if(value > penalty)
            return((value - penalty) / ( denom * (1.0 + l2 - 1.0 / gamma) ));
        else if(value < -penalty)
            return((value + penalty) / ( denom * (1.0 + l2 - 1.0 / gamma) ));
        else
            return(0.0);
         */
    }


    void set_threshold_func()
    {
        if (penalty == "lasso")
        {
            thresh_func = &CoordGLMDense::soft_threshold;
        } else if (penalty == "mcp")
        {
            thresh_func = &CoordGLMDense::mcp_threshold;
        } else
        {
            thresh_func = &CoordGLMDense::scad_threshold;
        }
    }

    //void next_beta(Vector &res, VectorXi &eligible)
    void next_beta(SparseVector &res, SparseVectori &eligible)
    {

        int j;
        double grad;

        // now update intercept if necessary
        update_intercept();

        // if no penalty multiplication factors specified
        if (penalty_factor_size < 1)
        {
            for (InIterVeci i_(eligible); i_; ++i_)
            {
                int j = i_.index();
                double beta_prev = beta.coeff( j ); //beta(j);

                // surprisingly it's faster to calculate this on an iteration-basis
                // and not pre-calculate it within each newton iteration..
                if (Xsq(j) < 0) Xsq(j) = (datX.col(j).array().square() * W.array()).matrix().mean();

                grad = datX.col(j).dot(resid_cur) / double(nobs) + beta_prev * Xsq(j);

                threshval = thresh_func(grad, lambda, gamma, lambda_ridge, Xsq(j));

                //  apply param limits
                if (threshval < limits(1,j)) threshval = limits(1,j);
                if (threshval > limits(0,j)) threshval = limits(0,j);

                // update both residual and linear predictor
                // if the coefficient changes after thresholding.
                if (beta_prev != threshval)
                {

                    if (threshval != 0.0) threshval = 0.85 * threshval + 0.15 * beta_prev;

                    beta.coeffRef(j)    = threshval;

                    VectorXd delta_cur = (threshval - beta_prev) * datX.col(j);

                    xbeta_cur         += delta_cur;
                    resid_cur.array() -= delta_cur.array() * W.array();

                    // update eligible set if necessary
                    if (threshval != 0.0 && eligible_set.coeff(j) == 0) eligible_set.coeffRef(j) = 1;
                    //if (threshval == 0.0 && eligible_set(j) == 1 && beta_nz_prev(j) == 0) eligible_set(j) = 0;
                    //if (threshval == 0.0 && eligible_set.coeff(j) == 1) eligible_set.coeffRef(j) = 0;
                } else
                {
                    // here we only remove a variable from the eligible set
                    // if it's zero twice in a row
                    if (beta_prev == 0.0 && eligible_set.coeff(j) == 1)
                    {
                        eligible_set.coeffRef(j) = 0;
                    }
                }
            }
        } else //if penalty multiplication factors are used
        {
            for (InIterVeci i_(eligible); i_; ++i_)
            {
                int j = i_.index();
                double beta_prev = beta.coeff( j ); //beta(j);

                // surprisingly it's faster to calculate this on an iteration-basis
                // and not pre-calculate it within each newton iteration..
                if (Xsq(j) < 0) Xsq(j) = (datX.col(j).array().square() * W.array()).matrix().mean();

                grad = datX.col(j).dot(resid_cur) / double(nobs) + beta_prev * Xsq(j);

                threshval = thresh_func(grad, penalty_factor(j) * lambda, gamma, penalty_factor(j) * lambda_ridge, Xsq(j));

                //  apply param limits
                if (threshval < limits(1,j)) threshval = limits(1,j);
                if (threshval > limits(0,j)) threshval = limits(0,j);

                // update both residual and linear predictor
                // if the coefficient changes after thresholding.
                if (beta_prev != threshval)
                {

                    if (threshval != 0.0) threshval = 0.85 * threshval + 0.15 * beta_prev;

                    beta.coeffRef(j) = threshval;

                    VectorXd delta_cur = (threshval - beta_prev) * datX.col(j);

                    xbeta_cur         += delta_cur;
                    resid_cur.array() -= delta_cur.array() * W.array();

                    // update eligible set if necessary
                    if (threshval != 0.0 && eligible_set.coeff(j) == 0) eligible_set.coeffRef(j) = 1;
                    //if (threshval == 0.0 && eligible_set(j) == 1 && beta_nz_prev(j) == 0) eligible_set(j) = 0;
                    //if (threshval == 0.0 && eligible_set.coeff(j) == 1) eligible_set.coeffRef(j) = 0;
                } else
                {
                    // here we only remove a variable from the eligible set
                    // if it's zero twice in a row
                    if (beta_prev == 0.0 && eligible_set.coeff(j) == 1)
                    {
                        eligible_set.coeffRef(j) = 0;
                    }
                }
            }
        }

    }

    //void next_beta(Vector &res, VectorXi &eligible)
    void next_beta(SparseVector &res, VectorXi &eligible)
    {

        int j;
        double grad;

        // now update intercept if necessary
        update_intercept();


        // if no penalty multiplication factors specified
        if (penalty_factor_size < 1)
        {
            for (j = 0; j < nvars; ++j)
            {
                if (eligible(j))
                {
                    double beta_prev = beta.coeff( j ); //beta(j);

                    // surprisingly it's faster to calculate this on an iteration-basis
                    // and not pre-calculate it within each newton iteration..
                    if (Xsq(j) < 0) Xsq(j) = (datX.col(j).array().square() * W.array()).matrix().mean();

                    grad = datX.col(j).dot(resid_cur) / double(nobs) + beta_prev * Xsq(j);

                    threshval = thresh_func(grad, lambda, gamma, lambda_ridge, Xsq(j));

                    //  apply param limits
                    if (threshval < limits(1,j)) threshval = limits(1,j);
                    if (threshval > limits(0,j)) threshval = limits(0,j);

                    // update both residual and linear predictor
                    // if the coefficient changes after thresholding.
                    if (beta_prev != threshval)
                    {

                        if (threshval != 0.0) threshval = 0.85 * threshval + 0.15 * beta_prev;

                        beta.coeffRef(j)    = threshval;

                        VectorXd delta_cur = (threshval - beta_prev) * datX.col(j);

                        xbeta_cur         += delta_cur;
                        resid_cur.array() -= delta_cur.array() * W.array();

                        // update eligible set if necessary
                        if (threshval != 0.0 && eligible_set.coeff(j) == 0) eligible_set.coeffRef(j) = 1;
                        //if (threshval == 0.0 && eligible_set(j) == 1 && beta_nz_prev(j) == 0) eligible_set(j) = 0;
                        //if (threshval == 0.0 && eligible_set.coeff(j) == 1) eligible_set.coeffRef(j) = 0;
                    } else
                    {
                        // here we only remove a variable from the eligible set
                        // if it's zero twice in a row
                        if (beta_prev == 0.0 && eligible_set.coeff(j) == 1)
                        {
                            eligible_set.coeffRef(j) = 0;
                        }
                    }
                } // end eligible set check
            }
        } else //if penalty multiplication factors are used
        {
            for (j = 0; j < nvars; ++j)
            {
                if (eligible(j))
                {
                    double beta_prev = beta.coeff( j ); //beta(j);

                    // surprisingly it's faster to calculate this on an iteration-basis
                    // and not pre-calculate it within each newton iteration..
                    if (Xsq(j) < 0) Xsq(j) = (datX.col(j).array().square() * W.array()).matrix().mean();

                    grad = datX.col(j).dot(resid_cur) / double(nobs) + beta_prev * Xsq(j);

                    threshval = thresh_func(grad, penalty_factor(j) * lambda, gamma, penalty_factor(j) * lambda_ridge, Xsq(j));

                    //  apply param limits
                    if (threshval < limits(1,j)) threshval = limits(1,j);
                    if (threshval > limits(0,j)) threshval = limits(0,j);

                    // update both residual and linear predictor
                    // if the coefficient changes after thresholding.
                    if (beta_prev != threshval)
                    {

                        if (threshval != 0.0) threshval = 0.85 * threshval + 0.15 * beta_prev;

                        beta.coeffRef(j) = threshval;

                        VectorXd delta_cur = (threshval - beta_prev) * datX.col(j);

                        xbeta_cur         += delta_cur;
                        resid_cur.array() -= delta_cur.array() * W.array();

                        // update eligible set if necessary
                        if (threshval != 0.0 && eligible_set.coeff(j) == 0) eligible_set.coeffRef(j) = 1;
                        //if (threshval == 0.0 && eligible_set(j) == 1 && beta_nz_prev(j) == 0) eligible_set(j) = 0;
                        //if (threshval == 0.0 && eligible_set.coeff(j) == 1) eligible_set.coeffRef(j) = 0;
                    } else
                    {
                        // here we only remove a variable from the eligible set
                        // if it's zero twice in a row
                        if (beta_prev == 0.0 && eligible_set.coeff(j) == 1)
                        {
                            eligible_set.coeffRef(j) = 0;
                        }
                    }
                } // end eligible set check
            }
        }

    }



    // Calculate ||v1 - v2||^2 when v1 and v2 are sparse
    static double diff_squared_norm(const SparseVector &v1, const SparseVector &v2)
    {
        const int n1 = v1.nonZeros(), n2 = v2.nonZeros();
        const double *v1_val = v1.valuePtr(), *v2_val = v2.valuePtr();
        const int *v1_ind = v1.innerIndexPtr(), *v2_ind = v2.innerIndexPtr();

        double r = 0.0;
        int i1 = 0, i2 = 0;
        while(i1 < n1 && i2 < n2)
        {
            if(v1_ind[i1] == v2_ind[i2])
            {
                double val = v1_val[i1] - v2_val[i2];
                r += val * val;
                i1++;
                i2++;
            } else if(v1_ind[i1] < v2_ind[i2]) {
                r += v1_val[i1] * v1_val[i1];
                i1++;
            } else {
                r += v2_val[i2] * v2_val[i2];
                i2++;
            }
        }
        while(i1 < n1)
        {
            r += v1_val[i1] * v1_val[i1];
            i1++;
        }
        while(i2 < n2)
        {
            r += v2_val[i2] * v2_val[i2];
            i2++;
        }

        return r;
    }


public:
    CoordGLMDense(ConstGenericMatrix &datX_,
                  ConstGenericVector &datY_,
                  ConstGenericVector &weights_,
                  ConstGenericVector &offset_,
                  ArrayXd &penalty_factor_,
                  ConstGenericMatrix &limits_,
                  std::string &penalty_,
                  Rcpp::Function var_,
                  Rcpp::Function mu_eta_,
                  Rcpp::Function linkinv_,
                  Rcpp::Function dev_resids_,
                  bool intercept_,
                  double alpha_      = 1.0,
                  double tol_        = 1e-6,
                  int    maxit_irls_ = 100,
                  double tol_irls_   = 1e-6) :
    CoordBase<Eigen::SparseVector<double> >
                (datX_.rows(), datX_.cols(), tol_),
                               datX(datX_.data(), datX_.rows(), datX_.cols()),
                               datY(datY_.data(), datY_.size()),
                               weights(weights_.data(), weights_.size()),
                               offset(offset_.data(), offset_.size()),
                               resid_cur(datX_.rows()),
                               xbeta_cur(datX_.rows()),
                               mu(datX_.rows()),
                               varmu(datX_.rows()),
                               mu_eta_nv(datX_.rows()),
                               W(datX_.rows()), weightssqrt(weights.array().sqrt()),
                               z(datX_.rows()),
                               penalty(penalty_),
                               penalty_factor(penalty_factor_),
                               var(var_), mu_eta(mu_eta_), linkinv(linkinv_), dev_resids(dev_resids_),
                               intercept(intercept_),
                               limits(limits_.data(), limits_.rows(), limits_.cols()),
                               alpha(alpha_), maxit_irls(maxit_irls_), tol_irls(tol_irls_),
                               penalty_factor_size(penalty_factor_.size()),
                               XY(datX.transpose() * (datY.array() * weights.array()).matrix()),
                               Xsq(datX_.cols())
    {}

    double get_lambda_zero()
    {
        if (penalty_factor_size > 0)
        {
            VectorXd XXtmp = datX.transpose().rowwise().sum();
            lambda0 = 0;
            for (int i = 0; i < penalty_factor.size(); ++i)
            {
                if (penalty_factor(i) != 0.0)
                {
                    double valcur = std::abs(XY(i)) / penalty_factor(i);

                    if (valcur > lambda0) lambda0 = valcur;
                }
            }
        } else
        {
            lambda0 = (XY).cwiseAbs().maxCoeff();
        }

        lambda0 /= ( alpha * 1.0 * double(nobs)); //std::pow(1e-6, 1.0/(99.0));

        return lambda0;
    }

    // init() is a cold start for the first lambda
    void init(double lambda_, double gamma_)
    {

        set_threshold_func();

        beta.setZero();

        lambda       = lambda_ * alpha;
        lambda_ridge = lambda_ * (1.0 - alpha);

        gamma        = gamma_;

        eligible_set.setZero();

        eligible_set.reserve(std::min(nobs, nvars));

        nzero = 0;

        deviance = 0.0;

        xbeta_cur.setZero();
        resid_cur.setZero();

        // this starts estimate of intercept
        initialize_params();

        double cutoff = 2.0 * lambda - lambda0;


        /*
        if (penalty_factor_size < 1)
        {
            for (int j = 0; j < nvars; ++j) if (std::abs(XY(j)) > (cutoff)) eligible_set.coeffRef(j) = 1;
        } else
        {
            for (int j = 0; j < nvars; ++j) if (std::abs(XY(j)) > (cutoff * penalty_factor(j))) eligible_set.coeffRef(j) = 1;
        }*/

        //beta.reserve( std::max(eligible_set.sum() + 10, std::min(nvars, nobs)) );

    }
    // when computing for the next lambda, we can use the
    // current main_x, aux_z, dual_y and rho as initial values
    void init_warm(double lambda_, double gamma_)
    {
        lprev        = lambda;
        lambda       = lambda_ * alpha;
        lambda_ridge = lambda_ * (1.0 - alpha);

        gamma        = gamma_;

        eligible_set.setZero();

        eligible_set.reserve(std::min(nobs, nvars));

        nzero = 0;

        deviance = 0.0;

        double cutoff = (2.0 * lambda - lprev);


        /*
        if (penalty_factor_size < 1)
        {
            for (int j = 0; j < nvars; ++j) if (std::abs(XY(j)) >  (cutoff)) eligible_set.coeffRef(j) = 1;
        } else
        {
            for (int j = 0; j < nvars; ++j) if (std::abs(XY(j)) > cutoff * penalty_factor(j)) eligible_set.coeffRef(j) = 1;
        }*/


        //beta.reserve( std::max(eligible_set.sum() + 10, std::min(nvars, nobs)) );
    }


    int solve(int maxit)
    {
        //int i;
        int irls_iter = 0;

        while(irls_iter < maxit_irls)
        {
            irls_iter++;

            beta_prev_irls = beta;
            deviance_prev = deviance;

            //xbeta_cur.array() = (datX * beta).array() + beta0; //this is efficient because beta is a sparse vector

            update_quadratic_approx();

            int current_iter = 0;

            // run once through all variables
            current_iter++;
            beta_prev = beta;
            ineligible_set.fill(1);

            update_beta(ineligible_set);

            while(current_iter < maxit)
            {
                while(current_iter < maxit)
                {
                    current_iter++;
                    beta_prev = beta;

                    update_quadratic_approx();

                    update_beta(eligible_set);

                    if(converged()) break;
                }

                current_iter++;
                beta_prev = beta;
                ineligible_set.fill(1);

                for (InIterVeci i_(eligible_set); i_; ++i_)
                {
                    ineligible_set(i_.index()) = 0;
                }

                update_quadratic_approx();

                update_beta(ineligible_set);

                if(converged()) break;
            } //end coordinate descent loop

            if(converged_irls()) break;

        } //end irls loop


        /*
        for (int j = 0; j < nvars; ++j)
        {
            if (beta(j) != 0)
                ++nzero;
        }
         */

        // force zeros to be actual zeros
        beta.prune(0.0);

        nzero = beta.nonZeros();



        loss = compute_loss();

        // print_footer();

        return irls_iter;
    }

    virtual double get_intercept() { return beta0; }
};



#endif // COORDGAUSSIANDENSE_H
