# library for boston dataset

library(MASS)

# importing data

data("Boston")

# analysing the structure of the data

str(Boston)

summary(Boston)

# changing chas variable into factor variable or categorical variable

Boston$chas <- as.factor(Boston$chas)

# splitting data into train and test

library(caTools)

set.seed(564)

split <- sample.split(Boston, SplitRatio = .75)

train <- subset(Boston, split == T)
test <- subset(Boston, split == F)

# importing packages for model

library(caret)
library(glmnet)

# importing library for pararrel computing
# since I have 2 cores so I will put 2

library(doParallel)
cl <- makePSOCKcluster(2)
registerDoParallel(cl)

# setting up the cross validation with 10 and 6 repeats

trainControl <- trainControl(method = 'repeatedcv',
                             number = 10,
                             repeats = 6)

# model for the linear regression

modelLR <- train(medv ~ .,
                 data= train,
                 method = "lm",
                 trControl = trainControl)

# model for the ridge regression
# it's important in ridge regression alpha is 0

modelRR <- train(medv ~ .,
                 data= train,
                 method = "glmnet",
                 tuneGrid = expand.grid(alpha = 0,
                                        lambda = seq(.0001,1,length = 5)),
                 trControl = trainControl)

# model for the lasso regression
# it's important in ridge regression alpha is 1

modelLSR <- train(medv ~ .,
                 data= train,
                 method = "glmnet",
                 tuneGrid = expand.grid(alpha = 1,
                                        lambda = seq(.0001,1,length = 5)),
                 trControl = trainControl)

# model for the elastic regression
# it's important in ridge regression alpha and lamda between 0 and 1

modelER <- train(medv ~ .,
                 data= train,
                 method = "glmnet",
                 tuneGrid = expand.grid(alpha = seq(0,1,length = 10),
                                        lambda = seq(.0001,1,length = 5)),
                 trControl = trainControl)

# closing pararrel computing

stopCluster(cl)

# analysing the models



# Linear regression

summary(modelLR)
# Multiple R-squared:  0.7252
# RMSE 5.12, MAE 3.6

# plotting the importance of different variable

plot(varImp(modelLR, scale = F))

# plotting fitted values vs Residuals, quantiles vs standardized resuduals etc. 

plot(modelLR$finalModel)

# prediction for the linear model

predictLR <- predict(modelLR, test)
plot(predictLR)



# Ridge regression

summary(modelRR)
# Multiple R-squared:  0.708
# RMSE 5.08, MAE 3.58

# plotting the importance of different variable

plot(varImp(modelRR, scale = F))

# plot for how the ridge regression select the optimal value through CV

plot(modelRR)

# plotting of coffecient of the features againt lambda and Fraction deviance

plot(modelRR$finalModel, xvar = 'lambda', label = T)
plot(modelRR$finalModel, xvar = 'dev', label = T)

# prediction for the ridge model

predictRR <- predict(modelRR, test)

plot(predictRR)



# Lasso regression

summary(modelLSR)
# Multiple R-squared:  0.6712479
# RMSE 5.551385, MAE 3.898416

# plotting the importance of different variable

plot(varImp(modelLSR, scale = F))

# plot for how the ridge regression select the optimal value through CV

plot(modelRR)

# plotting of coffecient of the features againt lambda and Fraction deviance

plot(modelLSR$finalModel, xvar = 'lambda', label = T)
plot(modelLSR$finalModel, xvar = 'dev', label = T)

# prediction for the ridge model

predictLSR <- predict(modelLSR, test)

plot(predictLSR)




# Elastic regression

summary(modelER)
# Multiple R-squared:  0.7117238
# RMSE 5.06, MAE 3.58

# plotting alpha values to see optimize value selection

plot(modelER)

# plotting the importance of different variable

plot(varImp(modelLSR, scale = F))

# plotting of coffecient of the features againt lambda and Fraction deviance

plot(modelER$finalModel, xvar = 'lambda', label = T)
plot(modelER$finalModel, xvar = 'dev', label = T)

# plot for how the ridge regression select the optimal value through CV

plot(modelRR)

# prediction for the ridge model

predictER <- predict(modelLSR, test)

plot(predictER)





