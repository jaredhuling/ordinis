#ifndef COORDBASE_H
#define COORDBASE_H

#include <RcppEigen.h>
#include "utils.h"


template<typename VecTypeBeta>
class CoordBase
{
protected:

    typedef Eigen::SparseVector<int> SparseVectori;

    const int nvars;      // dimension of beta
    const int nobs;       // number of rows

    VecTypeBeta beta;        // parameters to be optimized
    VecTypeBeta beta_prev;   // auxiliary parameters
    VecTypeBeta beta_prev_irls;   // auxiliary parameters
    SparseVectori eligible_set;
    VectorXi ineligible_set;
    double loss;

    double tol;           // tolerance for convergence

    int nzero;

    //virtual void next_beta(VecTypeBeta &res, VectorXi &eligible) = 0;
    virtual void next_beta(VecTypeBeta &res, SparseVectori &eligible) = 0;
    virtual void next_beta(VecTypeBeta &res, VectorXi &eligible) = 0;

    virtual bool converged()
    {
        return (stopRule(beta, beta_prev, tol));
    }

    virtual bool converged_irls()
    {
        return (stopRule(beta, beta_prev_irls, tol));
    }


    void print_row(int iter)
    {
        const char sep = ' ';

        Rcpp::Rcout << std::left << std::setw(7)  << std::setfill(sep) << iter;
        Rcpp::Rcout << std::endl;
    }
    void print_footer()
    {
        const int width = 80;
        Rcpp::Rcout << std::string(width, '=') << std::endl << std::endl;
    }

public:
    CoordBase(int n_, int p_,
              double tol_ = 1e-6) :
    nvars(p_), nobs(n_),
    beta(p_), beta_prev(p_), beta_prev_irls(p_), // allocate space but do not set values
    eligible_set(p_), ineligible_set(p_),
    tol(tol_)
    {}

    virtual ~CoordBase() {}

    void update_beta(SparseVectori &eligible)
    {
        //VecTypeBeta newbeta(nvars);
        next_beta(beta, eligible);
        //beta.swap(newbeta);
    }

    void update_beta(VectorXi &eligible)
    {
        //VecTypeBeta newbeta(nvars);
        next_beta(beta, eligible);
        //beta.swap(newbeta);
    }

    int solve(int maxit)
    {
        int i;

        nzero = 0;

        for(i = 0; i < maxit; ++i)
        {
            beta_prev = beta;
            // old_y = dual_y;
            //std::copy(dual_y.data(), dual_y.data() + dim_dual, old_y.data());

            update_beta(eligible_set);

            // print_row(i);

            if(converged())
                break;

        }

        // print_footer();

        return i + 1;
    }

    virtual VecTypeBeta get_beta() { return beta; }
    virtual int get_nzero() {return nzero;}
    double get_intercept() { return 0.0; }
    virtual double get_loss() { return loss; }
};



#endif // COORDBASE_H

