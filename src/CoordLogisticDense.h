#ifndef COORDLOGISTICDENSE_H
#define COORDLOGISTICDENSE_H

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
class CoordLogisticDense: public CoordBase<Eigen::SparseVector<double> > //Eigen::SparseVector<double>
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

    Scalar lambda, lambda_ridge, gamma;  // L1 penalty

    double threshval;
    VectorXd resid_cur, xbeta_cur, p, W, weightssqrt, z;

    std::string penalty;
    ArrayXd penalty_factor;       // penalty multiplication factors
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
    typedef double (*thresh_func_ptr)(double &value, const double &penalty, const double &gamma, const double &denom);

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

        xbeta_cur.fill(beta0);
    }

    void update_quadratic_approx()
    {
        p = 1.0 / (1.0 + ((-1.0 * xbeta_cur.array()).exp()));

        W = weights.array() * p.array() * (1 - p.array());

        // make sure no weights are too small
        for (int k = 0; k < nobs; ++k)
        {
            if (W(k) < 1e-5)
            {
                W(k) = 1e-5;
            }
        }

        resid_cur = weights.array() * (datY.array() - p.array()); // + xbeta_cur.array() * W.array().sqrt();

        Xsq = (W.array().sqrt().matrix().asDiagonal() * datX).array().square().colwise().sum();

        weights_sum = W.sum();
    }

    void update_intercept()
    {

        if (intercept)
        {
            resids_sum = (W.array() * resid_cur.array()).matrix().sum();

            double beta0_delta = resids_sum / weights_sum;

            beta0             += beta0_delta;

            resid_cur.array() -= beta0_delta * W.array();

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

    static double soft_threshold(double &value, const double &penalty, const double &gamma, const double &denom)
    {
        if(value > penalty)
            return( (value - penalty) / denom );
        else if(value < -penalty)
            return( (value + penalty) / denom );
        else
            return(0);
    }

    static double mcp_threshold(double &value, const double &penalty, const double &gamma, const double &denom)
    {
        if (std::abs(value) > gamma * penalty)
            return(value / denom);
        else if(value > penalty)
            return((value - penalty) / (denom - 1.0 / gamma) );
        else if(value < -penalty)
            return((value + penalty) / (denom - 1.0 / gamma));
        else
            return(0);
    }


    void set_threshold_func()
    {
        if (penalty == "lasso")
        {
            thresh_func = &CoordLogisticDense::soft_threshold;
        } else if (penalty == "mcp")
        {
            thresh_func = &CoordLogisticDense::mcp_threshold;
        } else
        {
            thresh_func = &CoordLogisticDense::soft_threshold;
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
                grad = datX.col(j).dot(resid_cur) + beta_prev * Xsq(j);

                threshval = thresh_func(grad, lambda, gamma, 1.0) / (Xsq(j) + lambda_ridge);

                //  apply param limits
                if (threshval < limits(1,j)) threshval = limits(1,j);
                if (threshval > limits(0,j)) threshval = limits(0,j);

                // update residual if the coefficient changes after
                // thresholding.
                if (beta_prev != threshval)
                {
                    beta.coeffRef(j)    = threshval;

                    VectorXd delta_cur = (threshval - beta_prev) * datX.col(j);

                    xbeta_cur         += delta_cur;
                    resid_cur.array() -= delta_cur.array() * W.array();

                    // update eligible set if necessary
                    if (threshval != 0.0 && eligible_set.coeff(j) == 0) eligible_set.coeffRef(j) = 1;
                    //if (threshval == 0.0 && eligible_set(j) == 1 && beta_nz_prev(j) == 0) eligible_set(j) = 0;
                    if (threshval == 0.0 && eligible_set.coeff(j) == 1) eligible_set.coeffRef(j) = 0;
                }
            }
        } else //if penalty multiplication factors are used
        {
            for (InIterVeci i_(eligible); i_; ++i_)
            {
                int j = i_.index();
                double beta_prev = beta.coeff( j ); //beta(j);
                grad = datX.col(j).dot(resid_cur) + beta_prev * Xsq(j);

                threshval = thresh_func(grad, penalty_factor(j) * lambda, gamma, 1.0) / (Xsq(j) + penalty_factor(j) * lambda_ridge);

                //  apply param limits
                if (threshval < limits(1,j)) threshval = limits(1,j);
                if (threshval > limits(0,j)) threshval = limits(0,j);

                // update residual if the coefficient changes after
                // thresholding.
                if (beta_prev != threshval)
                {
                    beta.coeffRef(j) = threshval;

                    VectorXd delta_cur = (threshval - beta_prev) * datX.col(j);

                    xbeta_cur         += delta_cur;
                    resid_cur.array() -= delta_cur.array() * W.array();

                    // update eligible set if necessary
                    if (threshval != 0.0 && eligible_set.coeff(j) == 0) eligible_set.coeffRef(j) = 1;
                    //if (threshval == 0.0 && eligible_set(j) == 1 && beta_nz_prev(j) == 0) eligible_set(j) = 0;
                    if (threshval == 0.0 && eligible_set.coeff(j) == 1) eligible_set.coeffRef(j) = 0;
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
                    grad = datX.col(j).dot(resid_cur) + beta_prev * Xsq(j);

                    threshval = thresh_func(grad, lambda, gamma, 1.0) / (Xsq(j) + lambda_ridge);

                    //  apply param limits
                    if (threshval < limits(1,j)) threshval = limits(1,j);
                    if (threshval > limits(0,j)) threshval = limits(0,j);

                    // update residual if the coefficient changes after
                    // thresholding.
                    if (beta_prev != threshval)
                    {
                        beta.coeffRef(j)    = threshval;

                        VectorXd delta_cur = (threshval - beta_prev) * datX.col(j);

                        xbeta_cur         += delta_cur;
                        resid_cur.array() -= delta_cur.array() * W.array();

                        // update eligible set if necessary
                        if (threshval != 0.0 && eligible_set.coeff(j) == 0) eligible_set.coeffRef(j) = 1;
                        //if (threshval == 0.0 && eligible_set(j) == 1 && beta_nz_prev(j) == 0) eligible_set(j) = 0;
                        if (threshval == 0.0 && eligible_set.coeff(j) == 1) eligible_set.coeffRef(j) = 0;
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
                    grad = datX.col(j).dot(resid_cur) + beta_prev * Xsq(j);

                    threshval = thresh_func(grad, penalty_factor(j) * lambda, gamma, 1.0) / (Xsq(j) + penalty_factor(j) * lambda_ridge);

                    //  apply param limits
                    if (threshval < limits(1,j)) threshval = limits(1,j);
                    if (threshval > limits(0,j)) threshval = limits(0,j);

                    // update residual if the coefficient changes after
                    // thresholding.
                    if (beta_prev != threshval)
                    {
                        beta.coeffRef(j) = threshval;

                        VectorXd delta_cur = (threshval - beta_prev) * datX.col(j);

                        xbeta_cur         += delta_cur;
                        resid_cur.array() -= delta_cur.array() * W.array();

                        // update eligible set if necessary
                        if (threshval != 0.0 && eligible_set.coeff(j) == 0) eligible_set.coeffRef(j) = 1;
                        //if (threshval == 0.0 && eligible_set(j) == 1 && beta_nz_prev(j) == 0) eligible_set(j) = 0;
                        if (threshval == 0.0 && eligible_set.coeff(j) == 1) eligible_set.coeffRef(j) = 0;
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
    CoordLogisticDense(ConstGenericMatrix &datX_,
                       ConstGenericVector &datY_,
                       ConstGenericVector &weights_,
                       ArrayXd &penalty_factor_,
                       ConstGenericMatrix &limits_,
                       std::string &penalty_,
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
                               resid_cur(datX_.rows()),
                               xbeta_cur(datX_.rows()),
                               p(datX_.rows()),
                               W(datX_.rows()), weightssqrt(weights.array().sqrt()),
                               z(datX_.rows()),
                               penalty(penalty_),
                               penalty_factor(penalty_factor_),
                               intercept(intercept_),
                               limits(limits_.data(), limits_.rows(), limits_.cols()),
                               alpha(alpha_), maxit_irls(maxit_irls_), tol_irls(tol_irls_),
                               penalty_factor_size(penalty_factor_.size()),
                               XY(datX.transpose() * (datY.array() * weights.array().sqrt()).matrix()),
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

        lambda0 /= ( alpha * 1.0 ); //std::pow(1e-6, 1.0/(99.0));

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

        xbeta_cur.setZero();
        resid_cur.setZero();

        // this starts estimate of intercept
        initialize_params();

        double cutoff = 2.0 * lambda - lambda0;


        if (penalty_factor_size < 1)
        {
            for (int j = 0; j < nvars; ++j) if (std::abs(XY(j)) > (cutoff)) eligible_set.coeffRef(j) = 1;
        } else
        {
            for (int j = 0; j < nvars; ++j) if (std::abs(XY(j)) > (cutoff * penalty_factor(j))) eligible_set.coeffRef(j) = 1;
        }

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

        double cutoff = (2.0 * lambda - lprev);

        /*
        if (penalty_factor_size < 1)
        {
            for (int j = 0; j < nvars; ++j) if (std::abs(XY(j)) >  std::pow(nobs, 0.85) * (cutoff)) eligible_set(j) = 1;
        } else
        {
            for (int j = 0; j < nvars; ++j) if (std::abs(XY(j)) > (std::pow(nobs, 0.85) * cutoff * penalty_factor(j))) eligible_set(j) = 1;
        }
         */

        //beta.reserve( std::max(eligible_set.sum() + 10, std::min(nvars, nobs)) );
    }


    int solve(int maxit)
    {
        //int i;
        int irls_iter = 0;

        beta_prev_irls = beta;

        while(irls_iter < maxit_irls)
        {
            irls_iter++;

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

                    //update_quadratic_approx();

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

                //update_quadratic_approx();

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


        nzero = beta.nonZeros();



        loss = compute_loss();

        // print_footer();

        return irls_iter;
    }

    virtual double get_intercept() { return beta0; }
};



#endif // COORDGAUSSIANDENSE_H
