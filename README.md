
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
               alpha = 0.5)  # use elastic net with alpha = 0.5

plot(mod)
```

![](vignettes/unnamed-chunk-1-1.png)<!-- -->

``` r
## show likelihood
logLik(mod)
```

    ## 'log Lik.' -1961.27343, -1959.70758, -1957.78064, -1955.98712, -1954.32196, -1951.19549, -1944.98744, -1935.19633, -1924.11824, -1912.07429, -1899.77184, -1886.44678, -1870.58999, -1853.24340, -1835.23735, -1817.49862, -1799.96557, -1782.29136, -1764.37648, -1743.68267, -1721.75126, -1697.32649, -1671.59098, -1642.93083, -1613.07350, -1583.14321, -1553.25824, -1522.85208, -1491.26696, -1459.76332, -1426.93728, -1394.03113, -1360.36283, -1327.75026, -1297.30627, -1266.64269, -1236.87644, -1208.28840, -1180.85662, -1154.88888, -1128.73910, -1103.03715, -1077.94018, -1052.46242, -1026.51134, -1002.31756,  -980.13814,  -956.93818,  -934.67016,  -914.11048,  -893.75828,  -873.12352,  -854.43625,  -837.66461,  -822.38819,  -808.65508,  -796.01017,  -783.56122,  -772.65736,  -763.01701,  -754.50812,  -747.08125,  -740.40515,  -733.52272,  -726.08586,  -719.73779,  -713.16250,  -705.59366,  -697.14824,  -687.39746,  -675.65564,  -662.28375,  -648.46673,  -634.62802,  -619.61490,  -603.58678,  -585.79452,  -567.43043,  -548.98786,  -529.39154,  -509.07860,  -488.90789,  -466.55592,  -444.19628,  -418.84690,  -392.18110,  -367.38593,  -343.85096,  -316.28301,  -290.44179,  -260.51427,  -233.30512,  -210.96515,  -186.49744,  -165.44093,  -125.84650,  -111.95472,   -88.25681,   -65.04833,   -42.28648 (df=  3  4  4  4  4  6  9 12 13 14 14 17 19 20 22 22 23 23 25 27 30 30 32 34 34 34 34 35 35 36 37 38 38 38 38 39 40 40 40 41 42 42 43 44 44 44 45 46 46 46 47 47 47 47 47 47 48 48 48 48 48 48 50 52 52 52 56 57 63 66 74 83 87 93 99106118122132142145146144147151152148151149150156152154155154157158158166164)

``` r
## compute AIC
AIC(mod)
```

    ##   [1] 3928.5469 3927.4152 3923.5613 3919.9742 3916.6439 3914.3910 3907.9749
    ##   [8] 3894.3927 3874.2365 3852.1486 3827.5437 3806.8936 3779.1800 3746.4868
    ##  [15] 3714.4747 3678.9972 3645.9311 3610.5827 3578.7530 3541.3653 3503.5025
    ##  [22] 3454.6530 3407.1820 3353.8617 3294.1470 3234.2864 3174.5165 3115.7042
    ##  [29] 3052.5339 2991.5266 2927.8746 2864.0623 2796.7257 2731.5005 2670.6125
    ##  [36] 2611.2854 2553.7529 2496.5768 2441.7132 2391.7778 2341.4782 2290.0743
    ##  [43] 2241.8804 2192.9248 2141.0227 2092.6351 2050.2763 2005.8764 1961.3403
    ##  [50] 1920.2210 1881.5166 1840.2470 1802.8725 1769.3292 1738.7764 1711.3102
    ##  [57] 1688.0203 1663.1224 1641.3147 1622.0340 1605.0162 1590.1625 1580.8103
    ##  [64] 1571.0454 1556.1717 1543.4756 1538.3250 1525.1873 1520.2965 1506.7949
    ##  [71] 1499.3113 1490.5675 1470.9335 1455.2560 1437.2298 1419.1736 1407.5890
    ##  [78] 1378.8609 1361.9757 1342.7831 1308.1572 1269.8158 1221.1118 1182.3926
    ##  [85] 1139.6938 1088.3622 1030.7719  989.7019  930.5660  880.8836  833.0285
    ##  [92]  770.6102  729.9303  682.9949  638.8819  565.6930  539.9094  492.5136
    ##  [99]  462.0967  412.5730

``` r
## BIC
BIC(mod)
```

    ##   [1] 3941.191 3944.274 3940.420 3936.833 3933.502 3939.679 3945.906
    ##   [8] 3944.968 3929.026 3911.153 3886.548 3878.542 3859.258 3830.779
    ##  [15] 3807.196 3771.719 3742.867 3707.519 3684.118 3655.160 3629.941
    ##  [22] 3581.091 3542.049 3497.158 3437.444 3377.583 3317.813 3263.215
    ##  [29] 3200.045 3143.253 3083.815 3024.217 2956.881 2891.656 2830.768
    ##  [36] 2775.655 2722.337 2665.161 2610.298 2564.577 2518.492 2467.088
    ##  [43] 2423.109 2378.368 2326.465 2278.078 2239.934 2199.748 2155.212
    ##  [50] 2114.093 2079.603 2038.334 2000.959 1967.416 1936.863 1909.397
    ##  [57] 1890.322 1865.424 1843.616 1824.335 1807.317 1792.464 1791.541
    ##  [64] 1790.205 1775.331 1762.635 1774.343 1765.420 1785.817 1784.959
    ##  [71] 1811.192 1840.380 1837.604 1847.215 1854.476 1865.922 1904.913
    ##  [78] 1893.043 1918.304 1941.257 1919.275 1885.149 1828.015 1801.940
    ##  [85] 1776.100 1728.983 1654.534 1626.108 1558.543 1513.075 1490.507
    ##  [92] 1411.231 1378.980 1336.259 1287.932 1227.386 1205.818 1158.422
    ##  [99] 1161.722 1103.769

## Performance

### Lasso

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
    "cd[lasso]"     = {reso <- ordinis(x, y, lambda = lambdas, 
                                       tol = 1e-3)},
    times = 5
)
```

    ## Unit: seconds
    ##           expr       min        lq      mean    median        uq       max
    ##  glmnet[lasso]  8.005078  8.127884  8.414827  8.222354  8.300543  9.418275
    ##      cd[lasso] 12.336802 14.393515 14.474512 15.042374 15.179461 15.420407
    ##  neval
    ##      5
    ##      5

``` r
# difference of results
max(abs(coef(resg) - reso$beta))
```

    ## [1] 0.0008885108

``` r
microbenchmark(
    "glmnet[lasso]" = {resg <- glmnet(x, y, thresh = 1e-15,  # thresh must be very low for comparable precision
                                      lambda = lambdas)},
    "ordinis[lasso]"     = {reso <- ordinis(x, y, lambda = lambdas, 
                                            tol = 1e-3)},
    times = 5
)
```

    ## Unit: seconds
    ##            expr      min       lq     mean   median       uq      max
    ##   glmnet[lasso] 11.41045 11.85669 12.20048 12.16824 12.65118 12.91585
    ##  ordinis[lasso] 12.22299 12.46392 13.10814 12.62983 13.97541 14.24853
    ##  neval
    ##      5
    ##      5

``` r
# difference of results
max(abs(coef(resg) - reso$beta))
```

    ## [1] 1.404917e-05

### Lasso (ill-conditioned)

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
    "cd[lasso]"     = {reso <- ordinis(x, y, lambda = lambdas, 
                                       tol = 1e-5)},
    times = 5
)
```

    ## Unit: milliseconds
    ##           expr      min       lq     mean   median       uq      max neval
    ##  glmnet[lasso] 311.5128 314.5104 326.1985 315.8275 334.3311 354.8109     5
    ##      cd[lasso] 546.7995 582.3956 629.4913 589.9005 675.9828 752.3780     5

``` r
# difference of results
max(abs(coef(resg) - reso$beta))
```

    ## [1] 0.0002629703

``` r
microbenchmark(
    "glmnet[lasso]" = {resg <- glmnet(x, y, thresh = 1e-11,  # thresh must be very low for comparable precision
                                      lambda = lambdas)},
    "ordinis[lasso]"     = {reso <- ordinis(x, y, lambda = lambdas, 
                                            tol = 1e-5)},
    times = 5
)
```

    ## Unit: milliseconds
    ##            expr      min       lq     mean   median       uq      max
    ##   glmnet[lasso] 635.1141 677.8965 730.9088 696.6094 759.0872 885.8370
    ##  ordinis[lasso] 569.6637 586.4960 638.8125 628.4924 642.3400 767.0705
    ##  neval
    ##      5
    ##      5

``` r
# difference of results
max(abs(coef(resg) - reso$beta))
```

    ## [1] 2.454331e-05
