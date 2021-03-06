% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/ordinis_methods.R
\name{predict.cv.ordinis}
\alias{predict.cv.ordinis}
\title{Prediction function for fitted cross validation ordinis objects}
\usage{
\method{predict}{cv.ordinis}(object, newx, s = c("lambda.min",
  "lambda.1se"), ...)
}
\arguments{
\item{object}{fitted \code{"cv.ordinis"} model object}

\item{newx}{Matrix of new values for \code{x} at which predictions are to be made. Must be a matrix; can be sparse as in the
\code{CsparseMatrix} objects of the \pkg{Matrix} package
This argument is not used for \code{type = c("coefficients","nonzero")}}

\item{s}{Value(s) of the penalty parameter lambda at which predictions are required. Default is the entire sequence used to create
the model. For \code{predict.cv.ordinis()}, can also specify \code{"lambda.1se"} or \code{"lambda.min"} for best lambdas estimated by cross validation}

\item{...}{used to pass the other arguments for predict.ordinis}
}
\value{
An object depending on the type argument
}
\description{
Prediction function for fitted cross validation ordinis objects
}
\examples{
set.seed(123)
n.obs <- 1e4
n.vars <- 100
n.obs.test <- 1e3

true.beta <- c(runif(15, -0.5, 0.5), rep(0, n.vars - 15))

x <- matrix(rnorm(n.obs * n.vars), n.obs, n.vars)
y <- rnorm(n.obs, sd = 3) + x \%*\% true.beta
x.test <- matrix(rnorm(n.obs.test * n.vars), n.obs.test, n.vars)
y.test <- rnorm(n.obs.test, sd = 3) + x.test \%*\% true.beta

fit <- cv.ordinis(x = x, y = y,
                  gamma = 1.4,
                  nlambda = 10)


preds.best <- predict(fit, newx = x.test, type = "response")

apply(preds.best, 2, function(x) mean((y.test - x) ^ 2))

}
