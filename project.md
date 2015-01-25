# Machine Learning Course Project

### Background
Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset).  

### Purpose
*DATA SOURCE* : [Human Activity Recognition](The data for this project come from this source: http://groupware.les.inf.puc-rio.br/har)  
*DATA SET PATH* : project local path `./pml-training.csv`, `./pml-testing.csv`  

***The goal of this document is to predict the manner in which they did the exercise. This is the "classe" variable in data set.***

### Load data

```r
library(caret)
set.seed(1234)
na.values = c('NA', '#DIV/0!', '')
data = read.csv('pml-training.csv', na.strings = na.values)
```
Load data for training.


### Pre-Processing
1. Remove Unuable Predictors

```r
data = data[, -c(1, 3, 4, 5)]
```
Remove *index* and *timestamp* columns that are not usable.

1. Remove too many NA value predictors

```r
total = dim(data)[1]
flag <- sapply(data, function(x) return(sum(is.na(x)) < total/2))
data = data[, flag]
```
Some columns have too many NA values. They disturb calculation of training.

2. Remove near zero predictors

```r
nzv = nearZeroVar(data)
data = data[, -nzv]
```
Remove *Zero Values* and *Near Zero Values* that do not affect result.

3. Remove high correlated predictors

```r
dataKlasses = sapply(data, class)
numericData = data[, (dataKlasses == 'numeric' | dataKlasses == 'integer')]
data.cor = cor(na.omit(numericData))
data.high.cor = findCorrelation(data.cor)
removeColumns = names(numericData[, data.high.cor])
data = data[, !(names(data) %in% removeColumns)]
```
High correlative predictors exist. Remove one of that high correlated predictors

### Predict
1. Data Splitting

```r
inTrain = createDataPartition(data$classe, p=.7, list=F)
data.train = data[inTrain, ]
data.test = data[-inTrain, ]
```

2. Train

```r
modFit = train(classe ~., method="rf", data=data.train, trControl=trainControl(method='cv', number=3, repeats=3, allowParallel=TRUE), preProcess = c('center', 'scale', 'knnImpute'))
modFit
```

```
## Random Forest 
## 
## 13737 samples
##    47 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## Pre-processing: centered, scaled,  nearest neighbor imputation 
## Resampling: Cross-Validated (3 fold) 
## 
## Summary of sample sizes: 9157, 9157, 9160 
## 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy   Kappa      Accuracy SD   Kappa SD    
##    2    0.9911190  0.9887648  0.0006286900  0.0007950252
##   26    0.9958509  0.9947517  0.0009994694  0.0012644553
##   51    0.9935218  0.9918051  0.0028484041  0.0036038577
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 26.
```
Random Forest gives high accuracy, but it is slow, just control number of folds in K-fold cross-validation.  
And add more pre-process methods(centering, scaling and knn imputing)  

3. Predict

```r
pred = predict(modFit, newdata=data.test)
confs = confusionMatrix(pred, data.test$classe)
confs$table
```

```
##           Reference
## Prediction    A    B    C    D    E
##          A 1674    4    0    0    0
##          B    0 1133    3    0    0
##          C    0    2 1023    1    0
##          D    0    0    0  963    1
##          E    0    0    0    0 1081
```
Accuracy is ***99.813 %***.


### Quiz result

```r
test = read.csv('pml-testing.csv', na.strings = na.values)
result = predict(modFit, newdata=test)
result
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```
