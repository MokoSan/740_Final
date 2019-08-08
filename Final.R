set.seed(123) # For reproducibility

#######################
# Loading in the Data #
#######################

# Load in the data frame and get a perfunctory summary.
all_pulsar_data <- read.csv("./pulsar_stars.csv")

# Check to see if any data cleaning is required.
any_na <- length(which(is.na(all_pulsar_data))); any_na

# Filtering out just the non-pulsar data. 
non_pulsar_data <- all_pulsar_data[all_pulsar_data$target_class == 0,]

# Getting rid of the redundant columns.
non_pulsar_data$target_class <- NULL
non_pulsar_data$Standard.deviation.of.the.DM.SNR.curve <- NULL
non_pulsar_data$Excess.kurtosis.of.the.DM.SNR.curve <- NULL
non_pulsar_data$Skewness.of.the.DM.SNR.curve <- NULL

# Get an initial summary.
summary(non_pulsar_data)

#############################
# Exploratory Data Analysis #
#############################

# Scale Quantities #
####################

non_pulsar_data$Mean.of.the.integrated.profile <- scale(non_pulsar_data$Mean.of.the.integrated.profile)
non_pulsar_data$Skewness.of.the.integrated.profile <- scale(Skewness.of.the.integrated.profile)
non_pulsar_data$Excess.kurtosis.of.the.integrated.profile <- scale(Excess.kurtosis.of.the.integrated.profile)

# Attaching to the pertinent dataframe.
attach(non_pulsar_data)

# Check for skewness #
#####################

# For: Integrated Profile
hist(Mean.of.the.integrated.profile, 
     main = "Histogram of Mean of Integrated Profile")
# The mean seems normally distributed.

hist(Standard.deviation.of.the.integrated.profile, 
     main = "Histogram of Std. Deviation of Integrated Profile")
# The standard deviation seems slightly right skewed but still pretty normal.

hist(Skewness.of.the.integrated.profile,
     main = "Histogram of Skewness of the Integrated Profile")
# The skewness seems normal.

hist(Excess.kurtosis.of.the.integrated.profile,
     main = "Histogram of Excess Kurtosis of Integrated Profile")
# The excess kurtosis is somewhat right skewed but still normal.

# DM SNR Curve
hist(Mean.of.the.DM.SNR.curve,
     main = "Histogram of the DM SNR Curve")
# Extreme right skewness of the mean implies that we need to log-transform
summary(Mean.of.the.DM.SNR.curve)
non_pulsar_data$Mean.of.the.DM.SNR.curve <- log(non_pulsar_data$Mean.of.the.DM.SNR.curve)
non_pulsar_data$Mean.of.the.DM.SNR.curve <- scale(non_pulsar_data$Mean.of.the.DM.SNR.curve)
hist(Mean.of.the.DM.SNR.curve,
     main = "Histogram of the DM SNR Curve")

# Correlation Analysis #
########################
library(corrplot) # For correlation analysis.

non_pulsar.cor <- cor(non_pulsar_data); non_pulsar.cor
corrplot(non_pulsar.cor, 
         method = "number", 
         number.cex = 0.75, 
         tl.cex = 0.5,
         title = "Correlation - Non-Pulsar Stars")

# It can be noted that there seems to be a reasonably high negative and positive 
# correlation between the predictors.

# Plotting Predictors vs. Response #
####################################

plot( x = Mean.of.the.integrated.profile, 
      y = Mean.of.the.DM.SNR.curve )
title( main =  "Mean Integrated Profile vs. Mean DM SNR Curve",
       cex.main = 1 )
      
plot( x = Standard.deviation.of.the.integrated.profile, 
      y = Mean.of.the.DM.SNR.curve )
title( main =  "Std. Dev of Integrated Profile vs. Mean DM SNR Curve",
       cex.main = 1 )

plot( x = Skewness.of.the.integrated.profile, 
      y = Mean.of.the.DM.SNR.curve )
title( main =  "Skewness Integrated Profile vs. Mean DM SNR Curve",
       cex.main = 0.9 )

plot( x = Excess.kurtosis.of.the.integrated.profile,
      y = Mean.of.the.DM.SNR.curve )
title( main =  "Excess Kurtosis Integrated Profile vs. Mean DM SNR Curve",
       cex.main = 0.7 )

# All plots seem extremely non-linear and no relationship seems obvious.

############################################
# Model Creation, Selection and Validation #
############################################

# Since the proposed response variable is numeric, this implies we'll have to use some form of regression.

# We have choice between the following model types:
# 1. KNN-Regression
# 2. Linear Model
# 3. Penalized Regression
# 4. Weighted Regression
# 5. Correlated Errors
# 6. Trees - Regression
# 7. SVM - Regression
# 8. ANN - Regression

# ANN Models
library(nnet)

# Hyperparameters to tune here are:
# 1. size: Number of units in the hidden layer. Can be zero if there are skip-layer units.
# 2. maxit: Maximum number of iterations. Default 100.
ann.sizes <- c(1,2,3)
ann.sizes.n <- length(ann.sizes)

# SVM Models
library(e1071)

# Hyperparameters to tune here are:
# 1. cost:  Cost of constraints violation (default: 1)—it is the 'C'-constant of the regularization term in the Lagrange formulation.
# 2. gamma: Parameter needed for all kernels except linear (default: 1/(data dimension)).
# 3. kernel: The kernel used in training and predicting - this will be set to 'radial' because of the observed non-linearities.
svm.cost <- c(.001, .01, .1, 1, 5, 10, 100)
svm.gamma <- c(0.5, 1, 2, 3, 4)
svm.kernel <- 'radial'

nmodels <- ann.sizes.n + 1 # +1 is for the best tuned svm model.

# Double Cross Validation #
###########################

# Outer validation with k = 10
fulldata.out <- non_pulsar_data
k.out <- 10
n.out <- nrow(non_pulsar_data)
groups.out <- c(rep(1:k.out,floor(n.out/k.out))); if(floor(n.out/k.out) != (n.out/k.out)) groups.out = c(groups.out, 1:(n.out%%k.out))
cvgroups.out <- sample(groups.out,n.out)  #orders randomly, with seed (123) 

# Set up storage for predicted values from the double-cross-validation
allpredictedCV.out <- rep(NA,n.out)

# Set up storage to see what models are "best" on the inner loops
allbestmodels <- rep(NA,k.out)

# Loop through outer splits.
for(j in 1:k.out) {
  groupj.out <- cvgroups.out == j
  
  # Training Data: Outer Layer
  traindata.out <- non_pulsar_data[!groupj.out,]
  trainx.out <- model.matrix(Mean.of.the.DM.SNR.curve ~ ., data = traindata.out)
  trainy.out <- traindata.out[,5]
  
  # Validation Data: Outer Layer
  validdata.out <- non_pulsar_data[groupj.out,]
  validx.out <- model.matrix(Mean.of.the.DM.SNR.curve ~ ., data=validdata.out)
  validy.out <- validdata.out[,5]
  
  # Setting up for the Inner loop
  fulldata.in <- traindata.out
  
  # We begin setting up the model-fitting process to use notation that will be
  # useful later, "in"side a validation
  n.in <- nrow(fulldata.in)
  x.in <- model.matrix(Mean.of.the.DM.SNR.curve ~ ., data = fulldata.in)
  y.in <- fulldata.in[,5]
  
  # Number folds and groups for (inner) cross-validation for model-selection
  k.in <- 10 
  
  # produce list of group labels
  groups.in <- c(rep(1:k.in,floor(n.in/k.in))); if(floor(n.in/k.in) != (n.in/k.in)) groups.in = c(groups.in, 1:(n.in%%k.in))
  
  cvgroups.in <- sample(groups.in,n.in) 
  allmodelCV.in <- rep(NA,nmodels + 1) # place-holder for results
  
  allpredictedCV.in.ann <- matrix(rep(NA, n.in * ann.sizes.n),
                                  ncol = ann.sizes.n)
  # Cycle through all folds:  fit the model to training data, predict test data,
  # and store the (cross-validated) predicted values
  for (i in 1:k.in)  {
    train.in <- (cvgroups.in != i)
    test.in <- (cvgroups.in == i)
    
    # ANN
    for (m in 1:ann.sizes.n) {
      ann.in <- nnet(formula = Mean.of.the.DM.SNR.curve ~ .,
                     data = non_pulsar_data,
                     subset = train.in,
                     trace = FALSE,
                     linout = TRUE,
                     size = ann.sizes[m])
      allpredictedCV.in.ann[test.in,m] <- predict(ann.in, fulldata.in[test.in,])
    }
  }
  
  for(m in (1 : ann.sizes.n)) {
    allmodelCV.in[m] <- mean((allpredictedCV.in.ann[,m]-fulldata.in$Mean.of.the.DM.SNR.curve)^2)
  }
  
  svm.tune.out <- tune(svm, 
                       Mean.of.the.DM.SNR.curve ~ ., 
                       data = fulldata.in, 
                       kernel = "radial", 
                       ranges = list(cost = svm.cost, gamma = svm.gamma ))
  allmodelCV.in[ann.sizes.n + 1] <- summary(svm.tune.out)$best.performance
  
  # compute and store the CV(10) values
  # visualize CV(10) values across all methods
  plot(allmodelCV.in,pch=20)
  
  # Get the best model.
  bestmodel.in <- (1:nmodels)[order(allmodelCV.in)[1]]  # actual selection
  
  # Fit the best model to the available data.
  # Best Model is an ANN Model.
  if (bestmodel.in <= ann.sizes.n) {
    bestsize <- ann.sizes[bestmodel.in]
    bestfit <- nnet(Mean.of.the.DM.SNR.curve ~ ., 
                    data = fulldata.in, 
                    size = bestsize, 
                    trace = FALSE,
                    linout = TRUE)
  }
  # Best Model is an SVM Model.
  else {
    bestcost <- svm.tune.out$best.parameters$cost
    bestfit <- svm(Mean.of.the.DM.SNR.curve ~ .,
                   data = fulldata.in,
                   cost = bestcost,
                   kernel = 'radial')
  }
  
  allbestmodels[j] = bestmodel.in
  # Predict using the best model 
  # Best Model is an ANN Model.
  if (bestmodel.in <= ann.sizes.n) {
    allpredictedCV.out[groupj.out] <- predict(bestfit,validdata.out)
  }
  # Best Model is an SVM Model.
  else {
    allpredictedCV.out[groupj.out] <- predict(bestfit,validdata.out)
  }
}

print( allbestmodels )
y.out <- non_pulsar_data$Mean.of.the.DM.SNR.curve
CV.out <- sum((allpredictedCV.out-y.out)^2)/n.out; CV.out
R2.out <- 1-sum((allpredictedCV.out-y.out)^2)/sum((y.out-mean(y.out))^2); R2.out