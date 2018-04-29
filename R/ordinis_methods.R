## the code here is largely based on the code
## from the glmnet package (no reason to reinvent the wheel)

#' Prediction method for coord lasso fitted objects
#'
#' @param object fitted "ordinis" model object
#' @param newx Matrix of new values for \code{x} at which predictions are to be made. Must be a matrix; can be sparse as in the
#' \code{CsparseMatrix} objects of the \pkg{Matrix} package.
#' This argument is not used for \code{type=c("coefficients","nonzero")}
#' @param s Value(s) of the penalty parameter lambda at which predictions are required. Default is the entire sequence used to create
#' the model.
#' @param type Type of prediction required. \code{type = "link"} gives the linear predictors for the \code{"binomial"} model; for \code{"gaussian"} models it gives the fitted values.
#' \code{type = "response"} gives the fitted probabilities for \code{"binomial"}. \code{type = "coefficients"} computes the coefficients at the requested values for \code{s}.
#' \code{type = "class"} applies only to \code{"binomial"} and produces the class label corresponding to the maximum probability.
#' @param ... not used
#' @importFrom graphics abline abline axis matplot points segments
#' @importFrom methods as
#' @importFrom stats approx predict quantile runif weighted.mean
#' @return An object depending on the type argument
#' @method predict ordinis
#' @export
#' @examples
#' set.seed(123)
#' n.obs <- 1e4
#' n.vars <- 100
#' n.obs.test <- 1e3
#'
#' true.beta <- c(runif(15, -0.5, 0.5), rep(0, n.vars - 15))
#'
#' x <- matrix(rnorm(n.obs * n.vars), n.obs, n.vars)
#' y <- rnorm(n.obs, sd = 3) + x %*% true.beta
#' x.test <- matrix(rnorm(n.obs.test * n.vars), n.obs.test, n.vars)
#' y.test <- rnorm(n.obs.test, sd = 3) + x.test %*% true.beta
#'
#' fit <- ordinis(x = x, y = y, nlambda = 10)
#'
#' preds.lasso <- predict(fit, newx = x.test, type = "response")
#'
#' apply(preds.lasso, 2, function(x) mean((y.test - x) ^ 2))
#'
predict.ordinis <- function(object, newx, s = NULL,
                            type = c("link",
                                     "response",
                                     "coefficients",
                                     "nonzero",
                                     "class"), ...)
{
    type <- match.arg(type)

    if(missing(newx)){
        if(!match(type, c("coefficients", "nonzero"), FALSE))stop("A value for 'newx' must be supplied")
    }
    nbeta <- object$beta

    if(!is.null(s)){
        #vnames=dimnames(nbeta)[[1]]
        lambda <- object$lambda
        lamlist <- lambda.interp(object$lambda,s)
        nbeta <- nbeta[,lamlist$left,drop=FALSE]*lamlist$frac +nbeta[,lamlist$right,drop=FALSE]*(1-lamlist$frac)
        #dimnames(nbeta)=list(vnames,paste(seq(along=s)))
    }
    if (type == "coefficients") return(nbeta)
    if (type == "nonzero") {
        newbeta <- abs(as.matrix(object$beta)) > 0
        index <- 1:(dim(newbeta)[1])
        nzel <- function(x, index) if(any(x)) index[x] else NULL
        betaList <- apply(newbeta, 2, nzel, index)
        return(betaList)
    }

    newx <- as.matrix(newx)
    # add constant column if needed
    if (ncol(newx) < nrow(nbeta))
        newx <- cbind(rep(1, nrow(newx)), newx)

    as.matrix(newx %*% nbeta)
}


#' @export
predict.cdgaussian <- function(object, newx, s = NULL,
                               type = c("link",
                                        "response",
                                        "coefficients",
                                        "nonzero"), ...)
{
    NextMethod("predict")
}


#' @export
predict.cdbinomial <- function(object, newx, s=NULL,
                               type=c("link",
                                      "response",
                                      "coefficients",
                                      "class",
                                      "nonzero"), ...)
{
    type <- match.arg(type)
    nfit <- NextMethod("predict")
    switch(type,
           response={
               prob=exp(-nfit)
               1 / (1 + prob)
           },
           class={
               cnum=ifelse(nfit > 0, 2, 1)
               clet=object$classnames[cnum]
               if(is.matrix(cnum))clet=array(clet,dim(cnum),dimnames(cnum))
               clet
           },
           nfit
    )
}




cv.cdbinomial=function(outlist,lambda,x,y,foldid,type.measure,grouped,keep=FALSE){
    typenames=c(mse="Mean-Squared Error",mae="Mean Absolute Error",deviance="Binomial Deviance",auc="AUC",class="Misclassification Error")
    if(type.measure=="default")type.measure="deviance"
    if(!match(type.measure,c("mse","mae","deviance","auc","class"),FALSE)){
        warning("Only 'deviance', 'class', 'auc', 'mse' or 'mae'  available for binomial models; 'deviance' used")
        type.measure="deviance"
    }

    ###These are hard coded in the Fortran, so we do that here too
    prob_min=1e-5
    prob_max=1-prob_min
    ###Turn y into a matrix
    nc = dim(y)
    if (is.null(nc)) {
        y = as.factor(y)
        ntab = table(y)
        nc = as.integer(length(ntab))
        y = diag(nc)[as.numeric(y), ]
    }
    N=nrow(y)
    nfolds=max(foldid)
    if( (N/nfolds <10)&&type.measure=="auc"){
        warning("Too few (< 10) observations per fold for type.measure='auc' in cv.lognet; changed to type.measure='deviance'. Alternatively, use smaller value for nfolds",call.=FALSE)
        type.measure="deviance"
    }
    if( (N/nfolds <3)&&grouped){
        warning("Option grouped=FALSE enforced in cv.glmnet, since < 3 observations per fold",call.=FALSE)
        grouped=FALSE
    }


    #if(!is.null(offset)){
    #  is.offset=TRUE
    #  offset=drop(offset)
    #}else is.offset=FALSE
    predmat=matrix(NA,nrow(y),length(lambda))
    nlams=double(nfolds)
    for(i in seq(nfolds)){
        which=foldid==i
        fitobj=outlist[[i]]
        #if(is.offset)off_sub=offset[which]
        preds=predict(fitobj, newx = x[which,,drop=FALSE], type="response")
        nlami=length(outlist[[i]]$lam)
        predmat[which,seq(nlami)]=preds
        nlams[i]=nlami
    }
    ###If auc we behave differently
    if(type.measure=="auc"){
        cvraw=matrix(NA,nfolds,length(lambda))
        good=matrix(0,nfolds,length(lambda))
        for(i in seq(nfolds)){
            good[i,seq(nlams[i])]=1
            which=foldid==i
            for(j in seq(nlams[i])){
                #cvraw[i,j]=auc.mat(y[which,],predmat[which,j],weights[which])
                cvraw[i,j]=auc.mat(y[which,],predmat[which,j], rep(1, sum(which)))
            }
        }
        N=apply(good,2,sum)
        #weights=tapply(weights,foldid,sum)
    }
    else{
        ##extract weights and normalize to sum to 1
        #ywt=apply(y,1,sum)
        #y=y/ywt
        #weights=weights*ywt

        N=nrow(y) - apply(is.na(predmat),2,sum)
        cvraw=switch(type.measure,
                     "mse"=(y[,1]-(1-predmat))^2 +(y[,2]-predmat)^2,
                     "mae"=abs(y[,1]-(1-predmat)) +abs(y[,2]-predmat),
                     "deviance"= {
                         predmat=pmin(pmax(predmat,prob_min),prob_max)
                         lp=y[,1]*log(1-predmat)+y[,2]*log(predmat)
                         ly=log(y)
                         ly[y==0]=0
                         ly=drop((y*ly)%*%c(1,1))
                         2*(ly-lp)
                     },
                     "class"=y[,1]*(predmat>.5) +y[,2]*(predmat<=.5)
        )
        if(grouped){
            cvob=cvcompute(cvraw,rep(1, nrow(y)),foldid,nlams)
            cvraw=cvob$cvraw;weights=cvob$weights;N=cvob$N
        }
    }
    #cvm=apply(cvraw,2,weighted.mean,w=weights,na.rm=TRUE)
    cvm=apply(cvraw,2,mean,w=weights,na.rm=TRUE)
    #cvsd=sqrt(apply(scale(cvraw,cvm,FALSE)^2,2,weighted.mean,w=rep(1, nrow(y)),na.rm=TRUE)/(N-1))
    cvsd=sqrt(apply(scale(cvraw,cvm,FALSE)^2,2,mean,na.rm=TRUE)/(N-1))
    out=list(cvm=cvm,cvsd=cvsd,name=typenames[type.measure])
    if(keep)out$fit.preval=predmat
    out

}

cv.cdgaussian <- function(outlist,lambda,x,y,foldid,type.measure,grouped,keep=FALSE)
{
    typenames=c(deviance="Mean-Squared Error",mse="Mean-Squared Error",mae="Mean Absolute Error",auc="Area under (ROC) Curve")
    if(type.measure=="default")type.measure="mse"
    if(!match(type.measure,c("mse","mae","deviance","auc"),FALSE)){
        warning("Only 'mse', 'deviance' or 'mae'  available for Gaussian models; 'mse' used")
        type.measure="mse"
    }
    #if(!is.null(offset))y=y-drop(offset)
    predmat=matrix(NA,length(y),length(lambda))
    nfolds=max(foldid)
    nlams=double(nfolds)
    for(i in seq(nfolds)){
        which=foldid==i
        fitobj=outlist[[i]]
        #fitobj$offset=FALSE
        preds=predict(fitobj,x[which,,drop=FALSE], type="response")
        nlami=length(outlist[[i]]$lambda)
        predmat[which,seq(nlami)]=preds
        nlams[i]=nlami
    }

    if (type.measure == "auc")
    {
        cvraw = matrix(NA, nfolds, length(lambda))
        good = matrix(0, nfolds, length(lambda))
        for (i in seq(nfolds)) {
            good[i, seq(nlams[i])] = 1
            which = foldid == i
            for (j in seq(nlams[i])) {
                #cvraw[i, j] = auc.mat(y[which, ], predmat[which,
                #                                          j], weights[which])
                cvraw[i,j] <- auc(y[which],predmat[which,j], rep(1, sum(which)))
            }
        }
        N = apply(good, 2, sum)
    } else
    {
        N=length(y) - apply(is.na(predmat),2,sum)
        cvraw=switch(type.measure,
                     "mse"=(y-predmat)^2,
                     "deviance"=(y-predmat)^2,
                     "mae"=abs(y-predmat)
        )
        if( (length(y)/nfolds <3)&&grouped){
            warning("Option grouped=FALSE enforced in cv.glmnet, since < 3 observations per fold",call.=FALSE)
            grouped=FALSE
        }
        if(grouped){
            cvob=cvcompute(cvraw,rep(1, length(y)),foldid,nlams)
            cvraw=cvob$cvraw;weights=cvob$weights;N=cvob$N
        }
    }


    #cvm=apply(cvraw,2,weighted.mean,w=weights,na.rm=TRUE)
    cvm=apply(cvraw,2,mean,na.rm=TRUE)
    #cvsd=sqrt(apply(scale(cvraw,cvm,FALSE)^2,2,weighted.mean,w=weights,na.rm=TRUE)/(N-1))
    cvsd=sqrt(apply(scale(cvraw,cvm,FALSE)^2,2,mean,na.rm=TRUE)/(N-1))
    out=list(cvm=cvm,cvsd=cvsd,name=typenames[type.measure])
    if(keep)out$fit.preval=predmat
    out
}







#' Plot method for ordinis fitted objects
#'
#' @param x fitted "ordinis" model object or fitted "cv.ordinis" model object
#' @param xvar What is on the X-axis. \code{"penalty"} plots against the penalty value applied to the coefficients, \code{"lambda"} against the log-lambda sequence
#' @param labsize size of labels for variable names. If labsize = 0, then no variable names will be plotted
#' @param xlab label for x-axis
#' @param ylab label for y-axis
#' @param main main title for plot
#' @param xlim numeric vectors of length 2, giving the \code{x} and \code{y} coordinates ranges.
#' @param n.print scalar integer for the number of times along the regularization path to print the number
#' of nonzero coefficients. If set to a negative value, the number of nonzero coefficients will not be printed.
#' @param ... other graphical parameters for the plot
#' @rdname plot
#' @export
#' @examples
#' set.seed(123)
#' n.obs <- 100
#' n.vars <- 1000
#'
#' true.beta <- c(runif(5, 0.1, 1) * (2 * rbinom(5, 1, 0.5) - 1), rep(0, n.vars - 5))
#'
#' x <- matrix(rnorm(n.obs * n.vars), n.obs, n.vars)
#' y <- rnorm(n.obs, sd = 2) + x %*% true.beta
#'
#' fit <- ordinis(x = x, y = y, penalty = c("mcp"))
#'
#' plot(fit)
#'
plot.ordinis <- function(x,
                         xvar = c("loglambda", "lambda", "penalty"),
                         labsize = 0.6,
                         xlab = iname, ylab = NULL,
                         main = x$penalty,
                         xlim = NULL,
                         n.print = 10L,
                         ...)
{

    xvar <- match.arg(xvar)
    nbeta <- as.matrix(x$beta[-1,]) ## remove intercept
    remove <- apply(nbeta, 1, function(betas) all(betas == 0) )

    if (is.null(xlim))
    {
        switch(xvar,
               "norm" = {
                   index    <- apply(abs(nbeta), 2, sum)
                   iname    <- expression(L[1] * " Norm")
                   xlim     <- range(index)
                   approx.f <- 1
               },
               "lambda" = {
                   index    <- x$lambda
                   iname    <- expression(lambda)
                   xlim     <- rev(range(index))
                   approx.f <- 0
               },
               "loglambda" = {
                   index    <- log(x$lambda)
                   iname    <- expression(log(lambda))
                   xlim     <- rev(range(index))
                   approx.f <- 1
               }
        )
    } else
    {
        switch(xvar,
               "norm" = {
                   index    <- apply(abs(nbeta), 2, sum)
                   iname    <- expression(L[1] * " Norm")
                   approx.f <- 1
               },
               "lambda" = {
                   index    <- x$lambda
                   iname    <- expression(lambda)
                   approx.f <- 0
               },
               "loglambda" = {
                   index    <- log(x$lambda)
                   iname    <- expression(log(lambda))
                   approx.f <- 1
               }
        )
    }

    if (all(remove)) stop("All beta estimates are zero for all values of lambda. No plot returned.")


    cols <- rainbow(sum(!remove))

    ## create sequence that grabs one of ROYGBIV and repeats with
    ## an increment up the rainbow spectrum with each step from 1:7 on ROYGBIV
    n.cols <- 7L
    scramble.seq <- rep(((1:n.cols) - 1) * (length(cols) %/% (n.cols)) + 1, length(cols) %/% n.cols)[1:length(cols)] +
        (((0:(length(cols)-1)) %/% n.cols))

    scramble.seq[is.na(scramble.seq)] <- which(!(1:length(cols) %in% scramble.seq))
    colseq <- cols[scramble.seq]


    matplot(index, t(nbeta[!remove,,drop=FALSE]),
            lty = 1,
            xlab = xlab,
            ylab = "",
            col = colseq,
            xlim = xlim,
            type = 'l', ...)

    if (is.null(ylab))
    {
        mtext(expression(hat(beta)), side = 2, cex = par("cex"), line = 3, las = 1)
    } else
    {
        mtext(ylab, side = 2, cex = par("cex"), line = 3)
        ylab = ""
    }

    if (n.print >= 0)
    {
        atdf <- pretty(index, n = n.print)
        plotnz <- approx(x = index, y = x$nzero, xout = atdf, rule = 2, method = "constant", f = approx.f)$y
        axis(side=3, at = atdf, labels = plotnz, tick=FALSE, line=0, ...)
    }

    title(main, line = 2.5, ...)



    # Adjust the margins to make sure the labels fit
    labwidth <- ifelse(labsize > 0, max(strwidth(rownames(nbeta[!remove,]), "inches", labsize)), 0)
    margins <- par("mai")
    par("mai" = c(margins[1:3], max(margins[4], labwidth*1.4)))
    if ( labsize > 0 && !is.null(rownames(nbeta)) )
    {
        take <- which(!remove)
        for (i in 1:sum(!remove)) {
            j <- take[i]
            axis(4, at = nbeta[j, ncol(nbeta)], labels = rownames(nbeta)[j],
                 las=1, cex.axis=labsize, col.axis = colseq[i],
                 lty = (i - 1) %% 5 + 1, col = colseq[i], ...)
        }
    }
    par("mai" = margins)
}


#' Prediction function for fitted cross validation ordinis objects
#'
#' @param object fitted \code{"cv.ordinis"} model object
#' @param newx Matrix of new values for \code{x} at which predictions are to be made. Must be a matrix; can be sparse as in the
#' \code{CsparseMatrix} objects of the \pkg{Matrix} package
#' This argument is not used for \code{type = c("coefficients","nonzero")}
#' @param s Value(s) of the penalty parameter lambda at which predictions are required. Default is the entire sequence used to create
#' the model. For \code{predict.cv.ordinis()}, can also specify \code{"lambda.1se"} or \code{"lambda.min"} for best lambdas estimated by cross validation
#' @param ... used to pass the other arguments for predict.ordinis
#' @return An object depending on the type argument
#' @method predict cv.ordinis
#' @export
#' @examples
#' set.seed(123)
#' n.obs <- 1e4
#' n.vars <- 100
#' n.obs.test <- 1e3
#'
#' true.beta <- c(runif(15, -0.5, 0.5), rep(0, n.vars - 15))
#'
#' x <- matrix(rnorm(n.obs * n.vars), n.obs, n.vars)
#' y <- rnorm(n.obs, sd = 3) + x %*% true.beta
#' x.test <- matrix(rnorm(n.obs.test * n.vars), n.obs.test, n.vars)
#' y.test <- rnorm(n.obs.test, sd = 3) + x.test %*% true.beta
#'
#' fit <- cv.ordinis(x = x, y = y,
#'                   gamma = 1.4,
#'                   nlambda = 10)
#'
#'
#' preds.best <- predict(fit, newx = x.test, type = "response")
#'
#' apply(preds.best, 2, function(x) mean((y.test - x) ^ 2))
#'
predict.cv.ordinis <- function(object, newx,
                            s=c("lambda.min", "lambda.1se"), ...)
{
    if(is.numeric(s))lambda=s
    else
        if(is.character(s)){
            s=match.arg(s)
            lambda=object[[s]]
        }

    else stop("Invalid form for s")
    predict(object$ordinis.fit, newx, s=lambda, ...)
}



#' Plot method for fitted two mountains cv objects
#'
#' @param sign.lambda Either plot against log(lambda) (default) or its negative if \code{sign.lambda = -1}.
#' @rdname plot
#' @method plot cv.ordinis
#' @export
#' @examples
#' set.seed(123)
#' n.obs <- 100
#' n.vars <- 200
#'
#' true.beta <- c(runif(15, -0.5, 0.5), rep(0, n.vars - 15))
#'
#' x <- matrix(rnorm(n.obs * n.vars), n.obs, n.vars)
#' y <- rnorm(n.obs, sd = 3) + x %*% true.beta
#'
#' fit <- cv.ordinis(x = x, y = y, gamma = 1.4)
#'
#' plot(fit)
#'
plot.cv.ordinis <- function(x, sign.lambda = 1, ...)
{
    # modified from glmnet
    object = x

    main.txt <- "Two Mountains CV"

    xlab=expression(log(lambda))
    if(sign.lambda<0)xlab=paste("-",xlab,sep="")
    plot.args=list(x    = sign.lambda * log(object$lambda),
                   y    = object$cvm,
                   ylim = range(object$cvup, object$cvlo),
                   xlab = xlab,
                   ylab = object$name,
                   type = "n")
    new.args=list(...)
    if(length(new.args))plot.args[names(new.args)]=new.args
    do.call("plot", plot.args)
    error.bars(sign.lambda * log(object$lambda),
               object$cvup,
               object$cvlo, width = 0.005)
    points(sign.lambda*log(object$lambda), object$cvm, pch=20, col="dodgerblue")
    axis(side=3,at=sign.lambda*log(object$lambda),labels = paste(object$nzero), tick=FALSE, line=0, ...)
    abline(v = sign.lambda * log(object$lambda.min), lty=2, lwd = 2, col = "firebrick1")
    abline(v = sign.lambda * log(object$lambda.1se), lty=2, lwd = 2, col = "firebrick1")
    title(main.txt, line = 2.5, ...)
    invisible()
}



#' log likelihood function for fitted ordinis objects
#'
#' @param object fitted "ordinis" model object.
#' @param REML an optional logical value. If \code{TRUE} the
#' restricted log-likelihood is returned, else, if \code{FALSE},
#' the log-likelihood is returned. Defaults to \code{FALSE}.
#' @param ... not used
#' @rdname logLik
#' @export
#' @examples
#' set.seed(123)
#' n.obs <- 200
#' n.vars <- 500
#'
#' true.beta <- c(runif(15, -0.25, 0.25), rep(0, n.vars - 15))
#' x <- matrix(rnorm(n.obs * n.vars), n.obs, n.vars)
#' y <- rnorm(n.obs, sd = 3) + x %*% true.beta
#'
#' fit <- ordinis(x = x, y = y)
#'
#' logLik(fit)
#'
#'
logLik.ordinis <- function(object, REML = FALSE, ...) {
    # taken from ncvreg. Thanks to Patrick Breheny.
    n  <- as.numeric(object$nobs)
    df <- object$nzero + object$intercept

    if (object$family == "gaussian")
    {
        if (REML)
        {
            rdf <- n - df
        } else
        {
            rdf <- n
        }

        resid.ss <- object$loss
        logL <- -0.5 * n * (log(2 * pi) - log(rdf) + log(resid.ss)) - 0.5 * rdf
    } else if (object$family == "binomial")
    {
        logL <- -1 * object$loss
    } else if (object$family == "poisson")
    {
        stop("poisson not complete yet")
        #y <- object$y
        #ind <- y != 0
        #logL <- -object$loss + sum(y[ind] * log(y[ind])) - sum(y) - sum(lfactorial(y))
    } else if (object$family == "coxph")
    {
        logL <- -1e99
    }

    attr(logL,"df")   <- df
    attr(logL,"nobs") <- n
    class(logL) <- "logLik"
    logL
}
