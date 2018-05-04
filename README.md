
## Introduction to \`ordinis’

The ‘ordinis’ package provides computation for penalized regression
problems via coordinate descent. It is mostly for my own experimentation
at this stage, however it is fairly efficient and reliable.

Install using the **devtools** package:

``` r
devtools::install_github("jaredhuling/ordinis")
```

or by cloning and building

## Example

``` r
library(ordinis)

# compute the full solution path, n > p
set.seed(123)
n <- 500
p <- 50000
m <- 50
b <- matrix(c(runif(m), rep(0, p - m)))
x <- matrix(rnorm(n * p, sd = 3), n, p)
y <- drop(x %*% b) + rnorm(n)

mod <- ordinis(x, y, 
               penalty = "mcp",
               lower.limits = rep(0, p), # force all coefficients to be positive
               penalty.factor = c(0, 0, rep(1, p-2)), # don't penalize first two coefficients
               alpha = 0.95)  # use elastic net with alpha = 0.95

plot(mod)
```

![](vignettes/unnamed-chunk-1-1.png)<!-- -->

``` r
## show likelihood
logLik(mod)
```

    ## 'log Lik.' -1960.4981, -1959.7942, -1958.4955, -1957.2568, -1956.0766, -1954.9533, -1953.8853, -1952.3674, -1950.1291, -1945.7116, -1939.9338, -1932.9379, -1925.4526, -1917.7379, -1909.7029, -1901.5909, -1893.7027, -1885.2011, -1874.5999, -1863.6113, -1852.3128, -1840.6926, -1828.9064, -1817.2957, -1805.9049, -1794.2657, -1782.7004, -1771.0541, -1758.8450, -1744.8511, -1730.6794, -1715.8181, -1699.7919, -1683.1596, -1666.3203, -1647.3779, -1627.8234, -1608.2236, -1588.6072, -1569.0073, -1549.4624, -1529.6515, -1508.9413, -1488.3112, -1467.8201, -1446.9552, -1426.0467, -1404.6194, -1383.3848, -1361.5585, -1340.0356, -1318.9663, -1298.7914, -1279.5254, -1259.3910, -1239.3235, -1220.1175, -1201.6675, -1183.7428, -1166.4344, -1149.8871, -1133.5086, -1116.3535, -1099.6555, -1082.5834, -1065.4142, -1048.6729, -1031.4340, -1014.8887,  -999.1320,  -984.1856,  -969.5954,  -954.8587,  -940.2463,  -925.9616,  -912.3066,  -899.2996,  -885.3278,  -871.5722,  -858.5513,  -846.3830,  -835.1985,  -824.8354,  -815.1351,  -806.1005,  -797.7647,  -789.3363,  -781.0021,  -773.5596,  -766.7392,  -760.4534,  -754.7102,  -749.4411,  -744.6367,  -740.2854,  -735.9902,  -731.7368,  -727.1736,  -722.7904,  -718.6086 (df= 3 4 4 4 4 4 4 5 7 9111213131414141619192021222222232324262628303031323434343434343535353536373738383838383840404040404040414242434344444444444546464646464747474747474747474848484848484848484950515252)

``` r
## compute AIC
AIC(mod)
```

    ##   [1] 3926.996 3927.588 3924.991 3922.514 3920.153 3917.907 3915.771
    ##   [8] 3914.735 3914.258 3909.423 3901.868 3889.876 3876.905 3861.476
    ##  [15] 3847.406 3831.182 3815.405 3802.402 3787.200 3765.223 3744.626
    ##  [22] 3723.385 3701.813 3678.591 3655.810 3634.531 3611.401 3590.108
    ##  [29] 3569.690 3541.702 3517.359 3491.636 3459.584 3428.319 3396.641
    ##  [36] 3362.756 3323.647 3284.447 3245.214 3206.015 3166.925 3129.303
    ##  [43] 3087.883 3046.622 3005.640 2965.910 2926.093 2883.239 2842.770
    ##  [50] 2799.117 2756.071 2713.933 2673.583 2635.051 2598.782 2558.647
    ##  [57] 2520.235 2483.335 2447.486 2412.869 2379.774 2349.017 2316.707
    ##  [64] 2283.311 2251.167 2216.828 2185.346 2150.868 2117.777 2086.264
    ##  [71] 2056.371 2029.191 2001.717 1972.493 1943.923 1916.613 1890.599
    ##  [78] 1864.656 1837.144 1811.103 1786.766 1764.397 1743.671 1724.270
    ##  [85] 1706.201 1689.529 1674.673 1658.004 1643.119 1629.478 1616.907
    ##  [92] 1605.420 1594.882 1585.273 1576.571 1569.980 1563.474 1556.347
    ##  [99] 1549.581 1541.217

``` r
## BIC
BIC(mod)
```

    ##   [1] 3939.640 3944.447 3941.849 3939.372 3937.012 3934.765 3932.629
    ##   [8] 3935.808 3943.760 3947.355 3948.228 3940.451 3931.695 3916.266
    ##  [15] 3906.410 3890.186 3874.410 3869.836 3867.277 3845.300 3828.918
    ##  [22] 3811.892 3794.534 3771.313 3748.531 3731.467 3708.337 3691.259
    ##  [29] 3679.270 3651.282 3635.368 3618.074 3586.022 3558.972 3531.508
    ##  [36] 3506.052 3466.943 3427.744 3388.511 3349.311 3310.222 3276.814
    ##  [43] 3235.394 3194.134 3153.152 3117.636 3082.034 3039.179 3002.925
    ##  [50] 2959.272 2916.226 2874.088 2833.738 2795.206 2767.366 2727.231
    ##  [57] 2688.819 2651.919 2616.070 2581.453 2548.358 2521.816 2493.721
    ##  [64] 2460.325 2432.395 2398.057 2370.789 2336.311 2303.220 2271.707
    ##  [71] 2241.814 2218.848 2195.589 2166.365 2137.795 2110.485 2084.471
    ##  [78] 2062.742 2035.231 2009.189 1984.853 1962.484 1941.757 1922.357
    ##  [85] 1904.288 1887.616 1876.974 1860.305 1845.420 1831.780 1819.208
    ##  [92] 1807.722 1797.183 1787.575 1778.872 1776.496 1774.204 1771.292
    ##  [99] 1768.741 1760.377

## Performance

### Lasso (linear regression)

``` r
library(microbenchmark)
library(glmnet)

b <- matrix(c(runif(m, min = -1), rep(0, p - m)))
x <- matrix(rnorm(n * p, sd = 3), n, p)
y <- drop(x %*% b) + rnorm(n)

lambdas = glmnet(x, y)$lambda

microbenchmark(
    "glmnet[lasso]" = {resg <- glmnet(x, y, thresh = 1e-10,  # thresh must be very small 
                                      lambda = lambdas)},    # for comparable precision
    "ordinis[lasso]" = {reso <- ordinis(x, y, lambda = lambdas, 
                                       tol = 1e-3)},
    times = 5
)
```

    ## Unit: seconds
    ##            expr      min       lq     mean   median       uq      max
    ##   glmnet[lasso] 3.558337 3.572053 3.590948 3.600987 3.606164 3.617202
    ##  ordinis[lasso] 5.737124 5.929748 5.981113 5.967628 5.999605 6.271460
    ##  neval
    ##      5
    ##      5

``` r
# difference of results
max(abs(coef(resg) - reso$beta))
```

    ## [1] 0.0008824882

``` r
microbenchmark(
    "glmnet[lasso]" = {resg <- glmnet(x, y, thresh = 1e-15,  # thresh must be very low for comparable precision
                                      lambda = lambdas)},
    "ordinis[lasso]" = {reso <- ordinis(x, y, lambda = lambdas, 
                                            tol = 1e-3)},
    times = 5
)
```

    ## Unit: seconds
    ##            expr      min       lq     mean   median       uq      max
    ##   glmnet[lasso] 5.097843 5.159453 5.233510 5.263395 5.271643 5.375216
    ##  ordinis[lasso] 5.351743 5.432486 5.526047 5.580820 5.629814 5.635374
    ##  neval
    ##      5
    ##      5

``` r
# difference of results
max(abs(coef(resg) - reso$beta))
```

    ## [1] 1.404332e-05

### Lasso (logistic regression)

`glmnet` is clearly faster for logistic regression for the same
precision

``` r
library(MASS)

set.seed(123)
n <- 200
p <- 10000
m <- 20
b <- matrix(c(runif(m, min = -0.5, max = 0.5), rep(0, p - m)))
x <- matrix(rnorm(n * p, sd = 3), n, p)
y <- 1 *(drop(x %*% b) + rnorm(n) > 0)

lambdas = glmnet(x, y, family = "binomial", lambda.min.ratio = 0.02)$lambda

microbenchmark(
    "glmnet[lasso]" = {resg <- glmnet(x, y, family = "binomial",
                                      thresh = 1e-10,  
                                      lambda = lambdas)},    
    "ordinis[lasso]"     = {reso <- ordinis(x, y, family = "binomial", 
                                            lambda = lambdas, 
                                            tol = 1e-3, tol.irls = 1e-3)},
    times = 5
)
```

    ## Unit: milliseconds
    ##            expr       min        lq     mean    median        uq      max
    ##   glmnet[lasso]  346.7732  350.3684  356.549  354.6519  357.7764  373.175
    ##  ordinis[lasso] 1094.7151 1096.0955 1105.472 1098.3948 1099.2188 1138.938
    ##  neval
    ##      5
    ##      5

``` r
# difference of results
max(abs(coef(resg) - reso$beta))
```

    ## [1] 0.0003735867

``` r
microbenchmark(
    "glmnet[lasso]" = {resg <- glmnet(x, y, family = "binomial",
                                      thresh = 1e-15,  
                                      lambda = lambdas)},    
    "ordinis[lasso]"     = {reso <- ordinis(x, y, family = "binomial", 
                                            lambda = lambdas, 
                                            tol = 1e-3, tol.irls = 1e-3)},
    times = 5
)
```

    ## Unit: milliseconds
    ##            expr       min        lq      mean    median        uq
    ##   glmnet[lasso]  612.8831  623.2813  631.1298  626.8109  645.9676
    ##  ordinis[lasso] 1078.4774 1087.8824 1102.9708 1097.9502 1122.6082
    ##        max neval
    ##   646.7064     5
    ##  1127.9360     5

``` r
# difference of results
max(abs(coef(resg) - reso$beta))
```

    ## [1] 2.525457e-05

### Lasso (linear regression, ill-conditioned)

``` r
library(MASS)

set.seed(123)
n <- 500
p <- 1000
m <- 50
b <- matrix(c(runif(m, min = -1), rep(0, p - m)))
sig <- matrix(0.5, ncol=p,nrow=p); diag(sig) <- 1
x <- mvrnorm(n, mu=rep(0, p), Sigma = sig)
y <- drop(x %*% b) + rnorm(n)

lambdas = glmnet(x, y)$lambda[1:65]

microbenchmark(
    "glmnet[lasso]" = {resg <- glmnet(x, y, thresh = 1e-9,  # thresh must be very small 
                                      lambda = lambdas)},    # for comparable precision
    "ordinis[lasso]" = {reso <- ordinis(x, y, lambda = lambdas, 
                                       tol = 1e-5)},
    times = 5
)
```

    ## Unit: milliseconds
    ##            expr      min       lq     mean   median       uq      max
    ##   glmnet[lasso] 148.5056 151.6153 151.5632 152.3556 152.4774 152.8624
    ##  ordinis[lasso] 281.0966 282.4747 289.1466 291.0906 294.7935 296.2775
    ##  neval
    ##      5
    ##      5

``` r
# difference of results
max(abs(coef(resg) - reso$beta))
```

    ## [1] 0.000262974

``` r
microbenchmark(
    "glmnet[lasso]" = {resg <- glmnet(x, y, thresh = 1e-11,  # thresh must be very low for comparable precision
                                      lambda = lambdas)},
    "ordinis[lasso]" = {reso <- ordinis(x, y, lambda = lambdas, 
                                            tol = 1e-5)},
    times = 5
)
```

    ## Unit: milliseconds
    ##            expr      min       lq     mean   median       uq      max
    ##   glmnet[lasso] 300.0399 303.5542 308.1593 305.9435 311.3657 319.8933
    ##  ordinis[lasso] 278.8825 280.5231 287.1311 280.7189 292.0318 303.4989
    ##  neval
    ##      5
    ##      5

``` r
# difference of results
max(abs(coef(resg) - reso$beta))
```

    ## [1] 2.454023e-05

### Validity of solutions with various bells and whistles

Due to internal differences in standardization, we now compare with
`glmnet` when using observation weights, penalty scaling factors, and
parameter box constraints

``` r
set.seed(123)
n = 200
p = 1000
m <- 15
b = c(runif(m, min = -0.5, max = 0.5), rep(0, p - m))
x = (matrix(rnorm(n * p, sd = 3), n, p))
y = drop(x %*% b) + rnorm(n)
y2 <- 1 * (y > rnorm(n, mean = 0.5, sd = 3))


wts <- runif(nrow(x))
wts <- wts / mean(wts) # re-scale like glmnet does, so we can compare

penalty.factor <- rbinom(ncol(x), 1, 0.99) * runif(ncol(x)) * 5
penalty.factor <- (penalty.factor / sum(penalty.factor)) * ncol(x)  # re-scale like glmnet does, so we can compare

system.time(resb <- ordinis(x, y2, family = "binomial", tol = 1e-7, tol.irls = 1e-5,
                            penalty = "lasso",
                            alpha = 0.5,  #elastic net term
                            lower.limits = 0, upper.limits = 0.02, # box constraints on all parameters
                            standardize = FALSE, intercept = FALSE,
                            weights = wts, # observation weights
                            penalty.factor = penalty.factor)) # penalty scaling factors
```

    ##    user  system elapsed 
    ##   0.066   0.000   0.066

``` r
system.time(resg <- glmnet(x,y2, family = "binomial",
                           lambda = resb$lambda,
                           alpha = 0.5, #elastic net term
                           weights = wts, # observation weights
                           penalty.factor = penalty.factor, # penalty scaling factors
                           lower.limits = 0, upper.limits = 0.02, # box constraints on all parameters
                           standardize = FALSE, intercept = FALSE,
                           thresh = 1e-16))
```

    ##    user  system elapsed 
    ##   0.037   0.001   0.038

``` r
## compare solutions
max(abs(resb$beta[-1,] - resg$beta))
```

    ## [1] 3.611923e-09
