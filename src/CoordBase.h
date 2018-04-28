#ifndef COORDBASE_H
#define COORDBASE_H

#include <RcppEigen.h>
#include "utils.h"


template<typename VecTypeX>
class CoordBase
{
protected:

    const int nvars;      // dimension of beta
    const int nobs;       // number of rows

    VecTypeX beta;        // parameters to be optimized
    VecTypeX beta_prev;   // auxiliary parameters
    VectorXi eligible_set;
    VectorXi ineligible_set;
    double loss;

    double tol;           // tolerance for convergence

    int nzero;

    virtual void next_beta(VecTypeX &res, VectorXi &eligible) = 0;

    virtual bool converged()
    {
        return (stopRule(beta, beta_prev, tol));
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
    beta(p_), beta_prev(p_), // allocate space but do not set values
    eligible_set(p_), ineligible_set(p_),
    tol(tol_)
    {}

    virtual ~CoordBase() {}

    void update_beta(VectorXi &eligible)
    {
        //VecTypeX newbeta(nvars);
        next_beta(beta, eligible);
        //beta.swap(newbeta);
    }

    int solve(int maxit)
    {
        int i;

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

        for (int j = 0; j < nvars; ++j)
        {
            if (beta(j) != 0)
                ++nzero;
        }

        // print_footer();

        return i + 1;
    }

    virtual VecTypeX get_beta() { return beta; }
    virtual int get_nzero() {return nzero;}

    virtual double get_loss() { return loss; }
};



#endif // COORDBASEACTIVE_H

