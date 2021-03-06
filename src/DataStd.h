#ifndef DATASTD_H
#define DATASTD_H

#include <RcppEigen.h>
#include <Eigen/Core>

using Eigen::MatrixXd;

template <typename Double = double>
class DataStd
{
private:
    typedef Eigen::SparseVector<double> SparseVector;
    typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> Matrix;
    typedef Eigen::Matrix<double, Eigen::Dynamic, 1> Vector;
    typedef Eigen::Array <double, Eigen::Dynamic, 1> Array;
    typedef const Eigen::Ref<const Vector> ConstGenericVector;
    typedef Eigen::Ref<Array> ArrayRef;


    // flag - 0: standardize = FALSE, intercept = FALSE
    //             directly fit model
    // flag - 1: standardize = TRUE, intercept = FALSE
    //             scale x and y by their standard deviation
    // flag - 2: standardize = FALSE, intercept = TRUE
    //             center x, standardize y
    // flag - 3: standardize = TRUE, intercept = TRUE
    //             standardize x and y
    const int flag;

    const int n;
    const int p;

    double meanY;
    double scaleY;
    bool   glm;
    Array  meanX;
    Array  scaleX;

    static double sd_n(ConstGenericVector &v)
    {
        double mean = v.mean();
        Vector v_centered = v.array() - mean;

        return v_centered.norm() / std::sqrt(double(v.size()));
    }

    // spvec -> spvec / arr, elementwise
    static void elementwise_quot(SparseVector &spvec, Array &arr)
    {
        for(typename SparseVector::InnerIterator iter(spvec); iter; ++iter)
        {
            iter.valueRef() /= arr[iter.index()];
        }
    }

    // inner product of spvec and arr
    static double sparse_inner_product(SparseVector &spvec, Array &arr)
    {
        double res = 0.0;
        for(typename SparseVector::InnerIterator iter(spvec); iter; ++iter)
        {
            res += iter.value() * arr[iter.index()];
        }
        return res;
    }

public:
    DataStd(int n_, int p_, bool standardize, bool intercept, bool glm_ = false) :
        flag(int(standardize) + 2 * int(intercept)),
        n(n_),
        p(p_),
        meanY(0.0),
        scaleY(1.0),
        glm(glm_)
    {
        if(flag == 3 || flag == 2)
            meanX.resize(p);
        if(flag == 3 || flag == 1)
            scaleX.resize(p);
    }

    void standardize(MatrixXd &X, Vector &Y, MatrixXd &limits)
    {
        double n_invsqrt = 1.0 / std::sqrt(Double(n));

        // standardize Y
        switch(flag)
        {
            case 1:
                if (!glm)
                {
                    scaleY = sd_n(Y);
                    Y.array() /= scaleY;
                }
                break;
            case 2:
            case 3:
                if (!glm)
                {
                    meanY = Y.mean();
                    Y.array() -= meanY;
                    scaleY = Y.norm() * n_invsqrt;
                    Y.array() /= scaleY;
                }
                break;
            default:
                break;
        }

        // standardize X
        switch(flag)
        {
            case 1:
                for(int i = 0; i < p; i++)
                {
                    //X.col(i).array() *=  wts.array();
                    scaleX[i] = sd_n(X.col(i));
                    X.col(i).array() *= (1.0 / scaleX[i]);
                    limits.col(i).array() *= (scaleX[i]);
                }
                break;
            case 2:
                for(int i = 0; i < p; i++)
                {
                    //X.col(i).array() *=  wts.array();
                    meanX[i] = X.col(i).mean();
                    X.col(i).array() -= meanX[i];
                    //limits.col(i).array() -= meanX[i];
                }
                break;
            case 3:
                for(int i = 0; i < p; i++)
                {
                    /*meanX[i] = X.col(i).mean();
                    X.col(i).array() -= meanX[i];
                    scaleX[i] = X.col(i).norm() * n_invsqrt;
                    X.col(i).array() /= scaleX[i];*/
                    double *begin = &X(0, i);
                    double *end = begin + n;
                    //X.col(i).array() *=  wts.array();
                    meanX[i] = X.col(i).mean();
                    std::transform(begin, end, begin, std::bind2nd(std::minus<double>(), meanX[i]));
                    scaleX[i] = X.col(i).norm() * n_invsqrt;
                    std::transform(begin, end, begin, std::bind2nd(std::multiplies<double>(), 1.0 / scaleX[i]));

                    limits.col(i).array() *= scaleX[i];
                    //double *beginll = &limits(0, i);
                    //double *endll = begin + 2;
                    //X.col(i).array() *=  wts.array();
                    //std::transform(begin, endll, beginll, std::bind2nd(std::plus<double>(), meanX[i]));
                    //std::transform(begin, endll, beginll, std::bind2nd(std::multiplies<double>(), scaleX[i]));
                }
                break;
            default:
                break;
        }
    }

    void standardize(MatrixXd &X, Vector &Y, MatrixXd &limits, Vector &wts)
    {
        double n_invsqrt = 1.0 / std::sqrt(Double(n));

        Vector sqrt_wts = wts.array().sqrt();

        if (!glm)
        {
            Y.array() *= sqrt_wts.array();
        }



        // standardize Y
        switch(flag)
        {
        case 1:
            if (!glm)
            {
                scaleY = sd_n(Y);
                Y.array() /= scaleY;
            }
            break;
        case 2:
        case 3:
            if (!glm)
            {
                meanY = Y.mean();
                Y.array() -= meanY;
                scaleY = Y.norm() * n_invsqrt;
                Y.array() /= scaleY;
            }
            break;
        default:
            break;
        }

        // standardize X
        switch(flag)
        {
        case 1:
            for(int i = 0; i < p; i++)
            {
                X.col(i).array() *=  sqrt_wts.array();
                scaleX[i] = sd_n(X.col(i));
                X.col(i).array() *= (1.0 / scaleX[i]);
            }
            break;
        case 2:
            for(int i = 0; i < p; i++)
            {
                X.col(i).array() *=  sqrt_wts.array();
                meanX[i] = X.col(i).mean();
                X.col(i).array() -= meanX[i];
            }
            break;
        case 3:
            for(int i = 0; i < p; i++)
            {
                /*meanX[i] = X.col(i).mean();
                X.col(i).array() -= meanX[i];
                scaleX[i] = X.col(i).norm() * n_invsqrt;
                X.col(i).array() /= scaleX[i];*/
                double *begin = &X(0, i);
                double *end = begin + n;
                X.col(i).array() *=  sqrt_wts.array();
                meanX[i] = X.col(i).mean();
                std::transform(begin, end, begin, std::bind2nd(std::minus<double>(), meanX[i]));
                scaleX[i] = X.col(i).norm() * n_invsqrt;
                std::transform(begin, end, begin, std::bind2nd(std::multiplies<double>(), 1.0 / scaleX[i]));
            }
            break;
        default:
            break;
        }
    }

    void recover(double &beta0, ArrayRef coef)
    {
        switch(flag)
        {
            case 0:
                beta0 = 0.0;
                break;
            case 1:
                coef /= scaleX;
                if (!glm)
                {
                    beta0 = 0.0;
                    coef *= scaleY;
                }
                break;
            case 2:
                if (!glm)
                {
                    coef *= scaleY;
                    beta0 = meanY - (coef * meanX).sum();
                } else
                {
                    beta0 -= (coef * meanX).sum();
                }
                break;
            case 3:
                coef /= scaleX;
                if (!glm)
                {
                    coef *= scaleY;
                    beta0 = meanY - (coef * meanX).sum();
                } else
                {
                    beta0 -= (coef * meanX).sum();
                }
                break;
            default:
                break;
        }
    }

    void recover(double &beta0, SparseVector &coef)
    {
        switch(flag)
        {
            case 0:
                beta0 = 0.0;
                break;
            case 1:
                elementwise_quot(coef, scaleX);
                if (!glm)
                {
                    coef *= scaleY;
                    beta0 = 0.0;
                }
                break;
            case 2:
                if (!glm)
                {
                    coef *= scaleY;
                    beta0 = meanY - sparse_inner_product(coef, meanX);
                } else
                {
                    beta0 -= sparse_inner_product(coef, meanX);
                }
                break;
            case 3:
                elementwise_quot(coef, scaleX);
                if (!glm)
                {
                    coef *= scaleY;
                    beta0 = meanY - sparse_inner_product(coef, meanX);
                } else
                {
                    beta0 -= sparse_inner_product(coef, meanX);
                }
                break;
            default:
                break;
        }
    }

    double get_scaleY() { return scaleY; }
};



#endif // DATASTD_H
