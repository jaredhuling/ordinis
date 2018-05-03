
## Taken from Jerome Friedman, Trevor Hastie, Noah Simon, and Rob Tibshirani's package glmnet
## https://cran.r-project.org/web/packages/glmnet/index.html
lambda.interp=function(lambda,s){
    ### lambda is the index sequence that is produced by the model
    ### s is the new vector at which evaluations are required.
    ### the value is a vector of left and right indices, and a vector of fractions.
    ### the new values are interpolated bewteen the two using the fraction
    ### Note: lambda decreases. you take:
    ### sfrac*left+(1-sfrac*right)

    if(length(lambda)==1){# degenerate case of only one lambda
        nums=length(s)
        left=rep(1,nums)
        right=left
        sfrac=rep(1,nums)
    }
    else{
        s[s > max(lambda)] = max(lambda)
        s[s < min(lambda)] = min(lambda)
        k=length(lambda)
        sfrac <- (lambda[1]-s)/(lambda[1] - lambda[k])
        lambda <- (lambda[1] - lambda)/(lambda[1] - lambda[k])
        coord <- approx(lambda, seq(lambda), sfrac)$y
        left <- floor(coord)
        right <- ceiling(coord)
        sfrac=(sfrac-lambda[right])/(lambda[left] - lambda[right])
        sfrac[left==right]=1
    }
    list(left=left,right=right,frac=sfrac)
}

## Taken from Jerome Friedman, Trevor Hastie, Noah Simon, and Rob Tibshirani's package glmnet
## https://cran.r-project.org/web/packages/glmnet/index.html
nonzeroCoef = function (beta, bystep = FALSE)
{
    ### bystep = FALSE means which variables were ever nonzero
    ### bystep = TRUE means which variables are nonzero for each step
    nr=nrow(beta)
    if (nr == 1) {#degenerate case
        if (bystep)
            apply(beta, 2, function(x) if (abs(x) > 0)
                1
                else NULL)
        else {
            if (any(abs(beta) > 0))
                1
            else NULL
        }
    }
    else {
        beta=abs(beta)>0 # this is sparse
        which=seq(nr)
        ones=rep(1,ncol(beta))
        nz=as.vector((beta%*%ones)>0)
        which=which[nz]
        if (bystep) {
            if(length(which)>0){
                beta=as.matrix(beta[which,,drop=FALSE])
                nzel = function(x, which) if (any(x))
                    which[x]
                else NULL
                which=apply(beta, 2, nzel, which)
                if(!is.list(which))which=data.frame(which)# apply can return a matrix!!
                which
            }
            else{
                dn=dimnames(beta)[[2]]
                which=vector("list",length(dn))
                names(which)=dn
                which
            }

        }
        else which
    }
}

## Taken from Rahul Mazumder, Trevor Hastie, and Jerome Friedman's sparsenet package
## https://cran.r-project.org/web/packages/sparsenet/index.html
argmin=function(x){
    vx=as.vector(x)
    imax=order(vx)[1]
    if(!is.matrix(x))imax
    else{
        d=dim(x)
        c1=as.vector(outer(seq(d[1]),rep(1,d[2])))[imax]
        c2=as.vector(outer(rep(1,d[1]),seq(d[2])))[imax]
        c(c1,c2)

    }
}







# taken from glmnet
cvcompute=function(mat,weights,foldid,nlams){
    ###Computes the weighted mean and SD within folds, and hence the se of the mean
    wisum=tapply(weights,foldid,sum)
    nfolds=max(foldid)
    outmat=matrix(NA,nfolds,ncol(mat))
    good=matrix(0,nfolds,ncol(mat))
    mat[is.infinite(mat)]=NA#just in case some infinities crept in
    for(i in seq(nfolds)){
        mati=mat[foldid==i,,drop=FALSE]
        wi=weights[foldid==i]
        outmat[i,]=apply(mati,2,weighted.mean,w=wi,na.rm=TRUE)
        good[i,seq(nlams[i])]=1
    }
    N=apply(good,2,sum)
    list(cvraw=outmat,weights=wisum,N=N)
}

# taken from glmnet
getmin=function(lambda,cvm,cvsd){
    cvmin=min(cvm,na.rm=TRUE)
    idmin=cvm<=cvmin
    lambda.min=max(lambda[idmin],na.rm=TRUE)
    idmin=match(lambda.min,lambda)
    semin=(cvm+cvsd)[idmin]
    idmin=cvm<=semin
    lambda.1se=max(lambda[idmin],na.rm=TRUE)
    list(lambda.min=lambda.min,lambda.1se=lambda.1se)
}



# taken from glmnet
auc=function(y,prob,w){
    if(missing(w)){
        rprob=rank(prob)
        n1=sum(y);n0=length(y)-n1
        u=sum(rprob[y==1])-n1*(n1+1)/2
        exp(log(u) - log(n1) - log(n0))
    }
    else{
        rprob=runif(length(prob))
        op=order(prob,rprob)#randomize ties
        y=y[op]
        w=w[op]
        cw=cumsum(w)
        w1=w[y==1]
        cw1=cumsum(w1)
        wauc = log(sum(w1*(cw[y==1]-cw1)))
        sumw1 = cw1[length(cw1)]
        sumw2  = cw[length(cw)] - sumw1
        exp(wauc - log(sumw1) - log(sumw2))
    }
}

# taken from glmnet
auc.mat=function(y,prob,weights=rep(1,nrow(y))){
    Weights=as.vector(weights*y)
    ny=nrow(y)
    Y=rep(c(0,1),c(ny,ny))
    Prob=c(prob,prob)
    auc(Y,Prob,Weights)
}

# taken from glmnet
error.bars <- function(x, upper, lower, width = 0.02, ...)
{
    xlim <- range(x)
    barw <- diff(xlim) * width
    segments(x, upper, x, lower, col = 8, lty = 5, lwd = 0.5, ...)
    segments(x - barw, upper, x + barw, upper, col = "grey50", lwd = 1, ...)
    segments(x - barw, lower, x + barw, lower, col = "grey50", lwd = 1, ...)
    range(upper, lower)
}

objective_logistic <- function(beta, x, y, lambda, penalty.factor = rep(1, ncol(x)), intercept = FALSE, alpha = 1)
{
    if (intercept)
    {
        beta0 <- beta[1]
        beta  <- beta[-1]

        xbeta <- drop(x %*% beta) + beta0
    } else
    {
        xbeta <- drop(x %*% beta)
    }
    neglogLik <- (-sum(  y * xbeta  ) + sum( log1p(exp(xbeta)) )) / nrow(x)

    neglogLik + sum(abs(beta) * lambda * penalty.factor * alpha) + 0.5 * sum((beta) ^ 2 * lambda * penalty.factor * (1 - alpha))
}

objective_linear <- function(beta, x, y, lambda, penalty.factor = rep(1, ncol(x)), intercept = FALSE, alpha = 1,
                             penalty = c("lasso", "mcp", "scad"), gamma = 3.7)
{
    penalty <- match.arg(penalty)
    if (intercept)
    {
        beta0 <- beta[1]
        beta  <- beta[-1]

        xbeta <- drop(x %*% beta) + beta0
    } else
    {
        xbeta <- drop(x %*% beta)
    }
    sumsq <- 0.5 * sum(  (y - xbeta) ^ 2  ) / nrow(x)

    if (penalty == "lasso")
    {
        penalty.part <- sum(abs(beta) * lambda * penalty.factor * alpha)
    } else if (penalty == "mcp")
    {
        penalty.part <- 0
        for (j in 1:length(beta))
        {
            pen.cur <- penalty.factor[j] * lambda * alpha
            b <- beta[j]
            if (abs(b) <= gamma * pen.cur)
            {
                penalty.part <- penalty.part + pen.cur * abs(b) - b ^ 2 / (2 * gamma)
            } else
            {
                penalty.part <- penalty.part + 0.5 * (gamma) * pen.cur ^ 2
            }
        }
    } else
    {
        penalty.part <- 0
        for (j in 1:length(beta))
        {
            pen.cur <- penalty.factor[j] * lambda * alpha
            b <- beta[j]
            if (abs(b) <= pen.cur)
            {
                penalty.part <- penalty.part + pen.cur * abs(b)
            } else if (abs(b) <= gamma * pen.cur)
            {
                penalty.part <- penalty.part - (abs(b) ^ 2 - 2 * gamma * pen.cur * abs(b) + pen.cur ^ 2) / (2 * (gamma - 1))
            } else
            {
                penalty.part <- penalty.part + 0.5 * (gamma + 1) * pen.cur ^ 2
            }
        }
    }

    sumsq + penalty.part + 0.5 * sum((beta) ^ 2 * lambda * penalty.factor * (1 - alpha))
}
