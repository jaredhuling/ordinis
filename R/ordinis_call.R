

#' Fitting Lasso-penalized Using the Coordinate Descent Algorithm
#'
#' @description ordinis provides estimation of linear models with the lasso penalty
#'
#'
#' @param x The design matrix
#' @param y The response vector
#' @param weights a vector of weights of length equal to length of \code{y}
#' @param penalty a string indicating which penalty to use. \code{"lasso"}, \code{"MCP"}, and \code{"SCAD"}
#' are available
#' @param lambda A user provided sequence of \eqn{\lambda}. If set to
#'                      \code{NULL}, the program will calculate its own sequence
#'                      according to \code{nlambda} and \code{lambda_min_ratio},
#'                      which starts from \eqn{\lambda_0} (with this
#'                      \eqn{\lambda} all coefficients will be zero) and ends at
#'                      \code{lambda0 * lambda_min_ratio}, containing
#'                      \code{nlambda} values equally spaced in the log scale.
#'                      It is recommended to set this parameter to be \code{NULL}
#'                      (the default).
#' @param alpha mixing parameter between 0 and 1 for elastic net. \code{alpha=1} is for the lasso, \code{alpha=0} is for ridge
#' @param gamma parameter for MCP/SCAD
#' @param penalty.factor a vector with length equal to the number of columns in x to be multiplied by lambda. by default
#'                      it is a vector of 1s
#' @param upper.limits a vector of length \code{ncol(x)} of upper limits for each coefficient. Can be a single value, which will
#' then be applied for each coefficient. Must be non-negative.
#' @param lower.limits a vector of length \code{ncol(x)} of lower limits for each coefficient. Can be a single value, which will
#' then be applied for each coefficient. Cannot be greater than 0.
#' @param nlambda Number of values in the \eqn{\lambda} sequence. Only used
#'                       when the program calculates its own \eqn{\lambda}
#'                       (by setting \code{lambda = NULL}).
#' @param lambda.min.ratio Smallest value in the \eqn{\lambda} sequence
#'                                as a fraction of \eqn{\lambda_0}. See
#'                                the explanation of the \code{lambda}
#'                                argument. This parameter is only used when
#'                                the program calculates its own \eqn{\lambda}
#'                                (by setting \code{lambda = NULL}). The default
#'                                value is the same as \pkg{glmnet}: 0.0001 if
#'                                \code{nrow(x) >= ncol(x)} and 0.01 otherwise.
#' @param family family of underlying model. Only "gaussian" for continuous responses is available now
#' @param intercept Whether to fit an intercept in the model. Default is \code{TRUE}.
#' @param standardize Whether to standardize the design matrix before
#'                    fitting the model. Default is \code{TRUE}. Fitted coefficients
#'                    are always returned on the original scale.
#' @param maxit Maximum number of coordinate descent iterations.
#' @param tol convergence tolerance parameter.
#' @param maxit.irls Maximum number of coordinate descent iterations.
#' @param tol.irls convergence tolerance parameter.
#'
#' @examples
#' set.seed(123)
#' n = 100
#' p = 1000
#' b = c(runif(10, min = 0.1, max = 1), rep(0, p - 10))
#' x = matrix(rnorm(n * p, sd = 3), n, p)
#' y = drop(x %*% b) + rnorm(n)
#'
#'
#' ## fit lasso model with 100 tuning parameter values
#' res <- ordinis(x, y)
#'
#' y2 <- 1 * (y > 0)
#'
#' resb <- ordinis(x, y2, family = "binomial")
#'
#' @export
ordinis <- function(x,
                    y,
                    weights          = rep(1, NROW(y)),
                    family           = c("gaussian", "binomial"),
                    penalty          = c("lasso", "mcp", "scad"),
                    lambda           = numeric(0),
                    alpha            = 1,
                    gamma            = 3.7,
                    penalty.factor   = NULL,
                    upper.limits     = rep(Inf, NCOL(x)),
                    lower.limits     = rep(-Inf, NCOL(x)),
                    nlambda          = 100L,
                    lambda.min.ratio = NULL,
                    intercept        = TRUE,
                    standardize      = TRUE,
                    maxit            = 5000L,
                    tol              = 1e-4,
                    maxit.irls       = 100L,
                    tol.irls         = 1e-4
)
{

    x <- as.matrix(x)
    y <- as.numeric(y)

    n <- nrow(x)
    p <- nvars <- ncol(x)

    intercept    <- as.logical(intercept)
    standardize  <- as.logical(standardize)
    family       <- match.arg(family)
    penalty      <- match.arg(penalty)

    if (!is.null(dim(x)))
    {
        vnames <- colnames(x)

        if (is.null(vnames))
        {
            vnames <- paste0("V", 1:p)
        }
    }

    if (family == "binomial")
    {
        if (length(unique(y)) != 2) stop("y must only take 2 values")
    }

    if (penalty == "scad") warning("scad not implemented yet, reverting to lasso")

    ## taken from glmnet
    if(any(lower.limits>0)) { stop("Lower limits should be non-positive") }
    if(any(upper.limits<0)) { stop("Upper limits should be non-negative") }
    lower.limits[lower.limits == -Inf] <- -1e99
    upper.limits[upper.limits == Inf]  <- 1e99
    if(length(lower.limits) < nvars)
    {
        if(length(lower.limits)==1) lower.limits <- rep(lower.limits, nvars) else stop("Require length 1 or nvars lower.limits")
    } else
    {
        lower.limits <- lower.limits[seq(nvars)]
    }
    if(length(upper.limits) < nvars)
    {
        if(length(upper.limits)==1) upper.limits <- rep(upper.limits, nvars) else stop("Require length 1 or nvars upper.limits")
    } else
    {
        upper.limits <- upper.limits[seq(nvars)]
    }

    if (n != NROW(y))
    {
        stop("number of rows in x not equal to length of y")
    }

    if (NROW(weights) != NROW(y))
    {
        stop("length of weights not equal to length of y")
    }

    if (is.null(penalty.factor))
    {
        penalty.factor <- numeric(0)
    } else
    {
        if (length(penalty.factor) != p)
        {
            stop("penalty.factor must be same length as number of columns in x")
        }
    }

    lambda_val <- sort(as.numeric(lambda), decreasing = TRUE)

    if(any(lambda_val <= 0))
    {
        stop("lambda must be positive")
    }


    if(nlambda[1] <= 0)
    {
        stop("nlambda must be a positive integer")
    }

    if(is.null(lambda.min.ratio))
    {
        lmr_val <- ifelse(nrow(x) < ncol(x), 0.01, 0.0001)
    } else
    {
        lmr_val <- as.numeric(lambda.min.ratio)
    }

    if(lmr_val >= 1 | lmr_val <= 0)
    {
        stop("lambda.min.ratio must be within (0, 1)")
    }

    lambda           <- lambda_val
    nlambda          <- as.integer(nlambda[1])
    lambda.min.ratio <- lmr_val

    if (alpha > 1 | alpha < 0) stop("alpha must be between 0 and 1")


    if(maxit <= 0)
    {
        stop("maxit should be positive")
    }
    if(tol < 0)
    {
        stop("tol should be nonnegative")
    }

    maxit       <- as.integer(maxit[1])
    tol         <- as.double(tol[1])
    maxit.irls  <- as.integer(maxit.irls[1])
    tol.irls    <- as.double(tol.irls[1])
    alpha       <- as.double(alpha[1])
    gamma       <- as.double(gamma[1])
    penalty     <- as.character(penalty[1])

    opts <- list(maxit       = maxit,
                 tol         = tol,
                 alpha       = alpha,
                 gamma       = gamma,
                 penalty     = penalty,
                 maxit.irls  = maxit.irls,
                 tol.irls    = tol.irls)

    if (gamma <= 1) stop("gamma must be greater than 1")

    if (family == "gaussian")
    {
        res <- coord_ordinis_dense_cpp(x,
                                       y,
                                       weights,
                                       lambda,
                                       penalty.factor,
                                       rbind(upper.limits, lower.limits),
                                       nlambda,
                                       lambda.min.ratio,
                                       standardize,
                                       intercept,
                                       opts
        )
        res$beta   <- res$beta[, 1:res$last, drop = FALSE]

        res$beta   <- as(res$beta, "sparseMatrix")

        res$fitted <- as.matrix(cbind(1, x) %*% res$beta)
        res$resid  <- matrix(rep(y, ncol(res$beta)), ncol = ncol(res$beta) ) - res$fitted
        res$loss   <- colSums(res$resid ^ 2)

    } else if (family == "binomial")
    {
        stop("binomial not yet supported")
        res <- coord_ordinis_dense_glm_cpp(x,
                                           y,
                                           weights,
                                           lambda,
                                           penalty.factor,
                                           rbind(upper.limits, lower.limits),
                                           nlambda,
                                           lambda.min.ratio,
                                           standardize,
                                           intercept,
                                           opts
        )
        res$beta   <- res$beta[, 1:res$last, drop = FALSE]

        res$beta   <- as(res$beta, "sparseMatrix")
    }

    rownames(res$beta) <- c("(Intercept)", vnames)

    res$niter  <- res$niter[1:res$last]
    res$lambda <- res$lambda[1:res$last]
    #res$losses <- res$losses[, 1:res$last, drop = FALSE]
    #res$losses.iter <- res$losses.iter[, 1:res$last, drop = FALSE]

    res$nzero   <- colSums(res$beta[-1,,drop=FALSE] != 0)


    res$family      <- family
    res$penalty     <- penalty
    res$standardize <- standardize
    res$intercept   <- intercept
    res$nobs        <- n
    res$nvars       <- p

    class2 <- switch(family,
                     "gaussian" = "cdgaussian",
                     "binomial" = "cdbinomial")

    class(res) <- c("ordinis", class2)
    res
}




#' CV Fitting for A Lasso Model Using the Coordinate Descent Algorithm
#'
#' @description Cross validation for linear models with the lasso penalty
#'
#' where \eqn{n} is the sample size and \eqn{\lambda} is a tuning
#' parameter that controls the sparsity of \eqn{\beta}.
#'
#' @param x The design matrix
#' @param y The response vector
#' @param lambda A user provided sequence of \eqn{\lambda}. If set to
#'                      \code{NULL}, the program will calculate its own sequence
#'                      according to \code{nlambda} and \code{lambda_min_ratio},
#'                      which starts from \eqn{\lambda_0} (with this
#'                      \eqn{\lambda} all coefficients will be zero) and ends at
#'                      \code{lambda0 * lambda_min_ratio}, containing
#'                      \code{nlambda} values equally spaced in the log scale.
#'                      It is recommended to set this parameter to be \code{NULL}
#'                      (the default).
#' @param gamma bandwidth for MCP/SCAD
#' @param type.measure measure to evaluate for cross-validation. The default is \code{type.measure = "deviance"},
#' which uses squared-error for gaussian models (a.k.a \code{type.measure = "mse"} there), deviance for logistic
#' regression. \code{type.measure = "class"} applies to binomial only. \code{type.measure = "auc"} is for two-class logistic
#' regression only. \code{type.measure = "mse"} or \code{type.measure = "mae"} (mean absolute error) can be used by all models;
#' they measure the deviation from the fitted mean to the response.
#' @param nfolds number of folds for cross-validation. default is 10. 3 is smallest value allowed.
#' @param foldid an optional vector of values between 1 and nfold specifying which fold each observation belongs to.
#' @param grouped Like in \pkg{glmnet}, this is an experimental argument, with default \code{TRUE}, and can be ignored by most users.
#' For all models, this refers to computing nfolds separate statistics, and then using their mean and estimated standard
#' error to describe the CV curve. If \code{grouped = FALSE}, an error matrix is built up at the observation level from the
#' predictions from the \code{nfold} fits, and then summarized (does not apply to \code{type.measure = "auc"}).
#' @param keep If \code{keep = TRUE}, a prevalidated list of arrasy is returned containing fitted values for each observation
#' and each value of lambda for each model. This means these fits are computed with this observation and the rest of its
#' fold omitted. The folid vector is also returned. Default is \code{keep = FALSE}
#' @param parallel If TRUE, use parallel foreach to fit each fold. Must register parallel before hand, such as \pkg{doMC}.
#' @param ... other parameters to be passed to \code{"ordinis"} function
#'
#' @examples set.seed(123)
#' n = 100
#' p = 1000
#' b = c(runif(10, min = 0.2, max = 1), rep(0, p - 10))
#' x = matrix(rnorm(n * p, sd = 3), n, p)
#' y = drop(x %*% b) + rnorm(n)
#'
#' ## fit lasso model with 100 tuning parameter values
#' res <- cv.ordinis(x, y)
#'
#'
#' @export
cv.ordinis <- function(x,
                       y,
                       lambda   = numeric(0),
                       gamma    = 3.7,
                       type.measure = c("mse", "deviance", "class", "auc", "mae"),
                       nfolds   = 10,
                       foldid   = NULL,
                       grouped  = TRUE,
                       keep     = FALSE,
                       parallel = FALSE,
                       ...)
{
    if (missing(type.measure))
        type.measure = "default"
    else type.measure = match.arg(type.measure)
    if (length(lambda) == 1 && length(lambda) < 2)
        stop("Need more than one value of lambda for cv.ordinis")
    N = nrow(x)
    y = drop(y)

    two.call = match.call(expand.dots = TRUE)
    which = match(c("type.measure", "nfolds", "foldid"), names(two.call), FALSE)
    if (any(which))
        two.call = two.call[-which]
    two.call[[1]] = as.name("ordinis")
    two.object = ordinis(x,
                         y,
                         lambda = lambda,
                         gamma  = gamma, ...)
    two.object$call = two.call


    nz = two.object$nzero

    if (is.null(foldid))
        foldid = sample(rep(seq(nfolds), length = N))
    else nfolds = max(foldid)
    if (nfolds < 3)
        stop("nfolds must be bigger than 3; nfolds=10 recommended")
    outlist = as.list(seq(nfolds))
    if (parallel) {
        outlist = foreach(i = seq(nfolds), .packages = c("ordinis")) %dopar%
        {
            which = foldid == i
            if (is.matrix(y))
                y_sub = y[!which, ]
            else y_sub = y[!which]

            ordinis(x[!which, , drop = FALSE],
                    y_sub,
                    lambda = lambda,
                    gamma  = gamma,
                    ...)
        }
    }
    else {
        for (i in seq(nfolds)) {
            which = foldid == i
            if (is.matrix(y))
                y_sub = y[!which, ]
            else y_sub = y[!which]

            outlist[[i]] = ordinis(x[!which, , drop = FALSE],
                                   y_sub,
                                   lambda = lambda,
                                   gamma  = gamma, ...)
        }
    }

    min.idx <- min(min(sapply(outlist, function(obj) obj$last)), two.object$last)

    two.object$beta        <- two.object$beta[, 1:min.idx, drop = FALSE]
    two.object$niter       <- two.object$niter[1:min.idx]
    two.object$lambda      <- two.object$lambda[1:min.idx]
    two.object$losses      <- two.object$losses[1:min.idx]
    two.object$losses.iter <- two.object$losses.iter[, 1:min.idx, drop = FALSE]
    two.object$nzero       <- two.object$nzero[1:min.idx]

    nz <- two.object$nzero

    for (i in 1:length(outlist))
    {
        outlist[[i]]$beta        <- outlist[[i]]$beta[, 1:min.idx, drop = FALSE]
        outlist[[i]]$niter       <- outlist[[i]]$niter[1:min.idx]
        outlist[[i]]$lambda      <- outlist[[i]]$lambda[1:min.idx]
        outlist[[i]]$losses      <- outlist[[i]]$losses[1:min.idx]
        outlist[[i]]$losses.iter <- outlist[[i]]$losses.iter[, 1:min.idx, drop = FALSE]
        outlist[[i]]$nzero       <- outlist[[i]]$nzero[1:min.idx]
    }

    fun     <- paste("cv", class(two.object)[[2]], sep = ".")
    cvstuff <- do.call(fun,
                       list(outlist,
                            two.object$lambda,
                            x,
                            y,
                            foldid,
                            type.measure,
                            grouped,
                            keep))
    cvm     <- cvstuff$cvm
    cvsd    <- cvstuff$cvsd
    cvname  <- cvstuff$name

    out <- list(lambda      = two.object$lambda,
                cvm         = cvm,
                cvsd        = cvsd,
                cvup        = cvm + cvsd,
                cvlo        = cvm - cvsd,
                name        = cvname,
                nzero       = nz,
                ordinis.fit = two.object)
    if(keep)out=c(out,list(fit.preval=cvstuff$fit.preval,foldid=foldid))
    lamin=if(type.measure=="auc")getmin(two.object$lambda,-cvm,cvsd)
    else getmin(two.object$lambda,cvm,cvsd)
    obj=c(out,as.list(lamin))
    class(obj)="cv.ordinis"
    obj

}





