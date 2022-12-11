# Title: C5T3 - Develop Models to Predict Sentiment

#Updated:  12.10.2022


###############
# Project Notes
###############


# Clear console: CTRL + L


###############
# Housekeeping
###############


###############
# Load packages
###############
install.packages("Rtools")
install.packages("caret")
install.packages("corrplot")
install.packages("readr")
install.packages("mlbench")
install.packages("doParallel")
install.packages("reshape2")
install.packages("dplyr")
install.packages("arules")
install.packages("arulesViz")
install.packages("RMariaDB")
install.packages("lubridate")
install.packages("plotly")
install.packages("ggfortify")
install.packages("forecast")
install.packages("e1071")
install.packages("class")
install.packages("ISLR")
install.packages("kknn")
install.packages("rpart")
library(caret)
library(corrplot)
library(readr)
library(mlbench)
library(doParallel)
library(e1071)
library(gbm)
library(ggplot2)
library(writexl)
library(reshape2)
library(dplyr)
library(arules)
library(arulesViz)
library(RMariaDB)
library(lubridate)
library(plotly)
library(ggfortify)
library(forecast) 
library(e1071)
library(class)
library(kknn)
library(rpart)
#library(ISLR)

# Clear objects if necessary
rm(list = ls())

# get working directory
getwd()

# set working directory 
setwd("C:/Users/giniewic/OneDrive - HP Inc/Documents/Personal/UT Data Analytics Cert/Course 5/C5T3")

# see files in working directory
dir()

# Find how many cores are on your machine
detectCores() # Result = 12

# Create Cluster with desired number of cores. Don't use them all! Your computer is running other processes. 
cl <- makeCluster(6)

# Register Cluster
registerDoParallel(cl)

# Confirm how many cores are now "assigned" to R and RStudio
getDoParWorkers() # Result 6


#####################
# Parallel Processing
#####################


####################
# Import data
####################

##-- Load the dataset
iphoneOOB <- read.csv("iphone_smallmatrix_labeled_8d.csv")
galaxyOOB <- read.csv("galaxy_smallmatrix_labeled_9d.csv")


#########################
#########################
# iPhone Data - from POA
#########################
#########################

##################
# Evaluate data
##################

# View first/last obs/rows
head(iphoneOOB)
tail(iphoneOOB)
anyNA(iphoneOOB)
anyDuplicated(iphoneOOB)

colnames(iphoneOOB)
summary(iphoneOOB)
str(iphoneOOB)

# Plot distribution of dependent variable (iPhone sentiment)
plot_ly(iphoneOOB, x=~iphoneOOB$iphonesentiment, type='histogram')



#######################
# Feature selection
#######################

#######################
# Correlation analysis
#######################

# for regression problems, the below rules apply.
# 1) compare each IV to the DV, if cor > 0.95, remove
# 2) compare each pair of IVs, if cor > 0.90, remove the
#    IV that has the lowest cor to the DV. (see code
#    below for setting a threshold to automatically select
#    IVs that are highly correlated)

# for classification problems, the below rule applies.
# 1) compare each pair of IVs, if cor > 0.90, remove one
#    of the IVs. (see code below to do this programmatically)

options(max.print=1000000)
corrAll <- cor(iphoneOOB[,1:58])
corrhigh <- findCorrelation(corrAll, cutoff=0.9) 
corrhigh
colnames(iphoneOOB[corrhigh])

# [1] 29 24 56 34 21 31 51 46 16 57 55  6  5
  

# [1] "samsungdisneg" "samsungdispos" "googleperneg"  "samsungdisunc" "nokiacamunc"   "nokiadisneg"   "nokiaperunc"   "nokiaperneg"   "nokiacamneg"  
#[10] "iosperunc"     "iosperneg"     "ios"           "htcphone"   


# plot correlation matrix
corrplot(corrAll, method = "circle", main="Correlation of iPhone OOB Data")
corrplot(corrAll, order = "hclust", main="Correlation of iPhone OOB Data") # sorts based on level of collinearity


# Remove highly correlated variables 
iphoneCOR <- iphoneOOB[,-corrhigh]

#######################
# Near Zero Variance
#######################

# table to see features with zero variance 
nzvMetrics <- nearZeroVar(iphoneOOB, saveMetrics = TRUE)
nzvMetrics

# create index to easily remove features 
nzv <- nearZeroVar(iphoneOOB, saveMetrics = FALSE)
nzv

# create a new data set and remove near zero variance features
iphoneNZV <- iphoneOOB[,-nzv]
str(iphoneNZV)

################################
# Recursive Feature Elimination 
################################

# Let's sample the data before using RFE
set.seed(123)
iphoneSample <- iphoneOOB[sample(1:nrow(iphoneOOB), 1000, replace=FALSE),]

# Set up rfeControl with randomforest, repeated cross validation and no updates
ctrl <- rfeControl(functions = rfFuncs, 
                   method = "repeatedcv",
                   repeats = 5,
                   verbose = FALSE)

# Use rfe and omit the response variable (attribute 59 iphonesentiment) 
rfeResults <- rfe(iphoneSample[,1:58], 
                  iphoneSample$iphonesentiment, 
                  sizes=(1:58), 
                  rfeControl=ctrl)

# Get results
rfeResults
# iphone, googleandroid, iphonedispos, iphonedisneg, samsunggalaxy

# Plot results
plot(rfeResults, type=c("g", "o"))


# create new data set with rfe recommended features
iphoneRFE <- iphoneOOB[,predictors(rfeResults)]

# add the dependent variable to iphoneRFE
iphoneRFE$iphonesentiment <- iphoneOOB$iphonesentiment

# review outcome
str(iphoneRFE)

### Four Datasets:
### iphoneOOB
### iphoneCOR
### iphoneNZV
### iphoneRFE

# Convert DV to factor (since it's classification)
iphoneOOB$iphonesentiment <- as.factor(iphoneOOB$iphonesentiment)
iphoneCOR$iphonesentiment <- as.factor(iphoneCOR$iphonesentiment)
iphoneNZV$iphonesentiment <- as.factor(iphoneNZV$iphonesentiment)
iphoneRFE$iphonesentiment <- as.factor(iphoneRFE$iphonesentiment)



##################
# Train/test sets
##################

###-----OOB Dataset-----###

# 70% will be for training; 30% for testing 
# Set seed
set.seed(123)

inTrainingOOB <- createDataPartition(as.factor(iphoneOOB$iphonesentiment), p=0.7, list=FALSE)
dataTrainOOB <- iphoneOOB[inTrainingOOB,]
dataTestOOB <- iphoneOOB[-inTrainingOOB,]

#verify number of observations
nrow(dataTrainOOB) #9082
nrow(dataTestOOB) #3891
################
# Train control
################

# set cross validation -- splits into 10 for cross-validation
fitControl <- trainControl(method="repeatedcv", number=10, repeats=1)

###############
# Train models
###############

names(getModelInfo())

# ----- C5.0 Model ----- #
set.seed(123)
C50OOB <- train(iphonesentiment~., data=dataTrainOOB, method="C5.0", importance = T, trControl=fitControl)
C50OOB

#model  winnow  trials  Accuracy   Kappa    
#rules  FALSE    1      0.7721012  0.5572920
#rules  FALSE   10      0.7603238  0.5399142
#rules  FALSE   20      0.7603238  0.5399142
#rules   TRUE    1      0.7734225  0.5601200
#rules   TRUE   10      0.7604328  0.5393405
#rules   TRUE   20      0.7604328  0.5393405
#tree   FALSE    1      0.7729815  0.5592708
#tree   FALSE   10      0.7630753  0.5460562
#tree   FALSE   20      0.7630753  0.5460562
#tree    TRUE    1      0.7728707  0.5593231
#tree    TRUE   10      0.7603195  0.5406196
#tree    TRUE   20      0.7603195  0.5406196


# ----- Random Forest Model ----- #
set.seed(123)
rfOOB <- train(iphonesentiment~., data=dataTrainOOB, method="rf", importance = T, trControl=fitControl)
rfOOB

#mtry  Accuracy   Kappa    
#2    0.7018607  0.3759693
#30    0.7725431  0.5625417
#58    0.7636254  0.5496866


# ----- Support Vector Model ----- #
set.seed(123)

svmOOB <- svm(iphonesentiment~., data=dataTrainOOB)
svmOOB
summary(svmOOB)
confusionMatrix(dataTrainOOB$iphonesentiment, predict(svmOOB))

#Overall Statistics
#Accuracy : 0.7065         
#95% CI : (0.697, 0.7158)
#No Information Rate : 0.8437         
#P-Value [Acc > NIR] : 1              
#Kappa : 0.3995   

svmOOB2 <- train(iphonesentiment~., data=dataTrainOOB, method="svmLinear2", importance = T, trControl=fitControl)
svmOOB2

#cost  Accuracy   Kappa    
#0.25  0.7058294  0.4003344
#0.50  0.7067053  0.4060204
#1.00  0.7068155  0.4075574


# ----- Weighted K-Nearest Neighbors Model ----- #
set.seed(123)
kknnOOB <- train(iphonesentiment~., data=dataTrainOOB, method="kknn", trControl=fitControl, preProcess = c("center", "scale"), tuneLength = 10)
kknnOOB

#kmax  Accuracy   Kappa    
#5    0.3103589  0.1562627
#7    0.3232428  0.1600047
#9    0.3279768  0.1607681
#11    0.3399754  0.1683071
#13    0.3452608  0.1715001
#15    0.3484537  0.1726067
#17    0.3502150  0.1732079
#19    0.3577041  0.1775238
#21    0.3623275  0.1791875
#23    0.3673920  0.1825012


############################
# Predict testSet/validation
############################

# ----- predict with C5.0 Model ----- #
c50PredictOOB <- predict(C50OOB, dataTestOOB)
postResample(c50PredictOOB, dataTestOOB$iphonesentiment)

#Accuracy     Kappa 
#0.7724936 0.5558736 

# ----- predict with Random Forest Model ----- #
rfPredictOOB <- predict(rfOOB, dataTestOOB)
postResample(rfPredictOOB, dataTestOOB$iphonesentiment)

#Accuracy     Kappa 
#0.7735219 0.5608873 

# ----- predict with Support Vector Model ----- #
svmPredictOOB <- predict(svmOOB, dataTestOOB)
postResample(svmPredictOOB, dataTestOOB$iphonesentiment)

#Accuracy     Kappa 
#0.6930591 0.3694901 

svmPredictOOB2 <- predict(svmOOB2, dataTestOOB)
postResample(svmPredictOOB2, dataTestOOB$iphonesentiment)

#Accuracy     Kappa 
#0.7125964 0.4214205

# ----- predict with Weighted K-Nearest Neighbor Model ----- #
kknnPredictOOB <- predict(kknnOOB, dataTestOOB)
postResample(kknnPredictOOB, dataTestOOB$iphonesentiment)

#Accuracy     Kappa 
#0.3663239 0.1835758 


# Since RF and C5.0 are very similar - create confusion matrices
cmRF <- confusionMatrix(rfPredictOOB, dataTestOOB$iphonesentiment)
cmRF

cmc50 <- confusionMatrix(c50PredictOOB, dataTestOOB$iphonesentiment)
cmc50

### Most accurate for OOB is Random Forest ###


###-----Correlation Dataset-----###

# 70% will be for training; 30% for testing 
# Set seed
set.seed(123)

inTrainingCOR <- createDataPartition(as.factor(iphoneCOR$iphonesentiment), p=0.7, list=FALSE)
dataTrainCOR <- iphoneCOR[inTrainingCOR,]
dataTestCOR <- iphoneCOR[-inTrainingCOR,]

#verify number of observations
nrow(dataTrainCOR) #9083
nrow(dataTestCOR) #3890
################
# Train control
################

# set cross validation -- splits into 10 for cross-validation
fitControl <- trainControl(method="repeatedcv", number=10, repeats=1)

###############
# Train models
###############

### Only using C5.0 and RF, since they performed best with OOB
names(getModelInfo())

# ----- C5.0 Model ----- #
set.seed(123)
C50COR <- train(iphonesentiment~., data=dataTrainCOR, method="C5.0", importance = T, trControl=fitControl)
C50COR

#model  winnow  trials  Accuracy   Kappa    
#rules  FALSE    1      0.7744109  0.5624722
#rules  FALSE   10      0.7631813  0.5466776
#rules  FALSE   20      0.7631813  0.5466776
#rules   TRUE    1      0.7746311  0.5629058
#rules   TRUE   10      0.7640599  0.5465686
#rules   TRUE   20      0.7640599  0.5465686
#tree   FALSE    1      0.7736397  0.5619226
#tree   FALSE   10      0.7631823  0.5462455
#tree   FALSE   20      0.7631823  0.5462455
#tree    TRUE    1      0.7746302  0.5637972
#tree    TRUE   10      0.7609797  0.5426669
#tree    TRUE   20      0.7609797  0.5426669


# ----- Random Forest Model ----- #
set.seed(123)
rfCOR <- train(iphonesentiment~., data=dataTrainCOR, method="rf", importance = T, trControl=fitControl)
rfCOR

#mtry  Accuracy   Kappa    
#2    0.6934954  0.3517781
#23    0.7738614  0.5657275
#45    0.7660444  0.5557082

############################
# Predict testSet/validation
############################

# ----- predict with C5.0 Model ----- #
c50PredictCOR <- predict(C50COR, dataTestCOR)
postResample(c50PredictCOR, dataTestCOR$iphonesentiment)

#Accuracy     Kappa 
#0.7660668 0.5417684 

# ----- predict with Random Forest Model ----- #
rfPredictCOR <- predict(rfCOR, dataTestCOR)
postResample(rfPredictCOR, dataTestCOR$iphonesentiment)

#Accuracy     Kappa 
#0.7694087 0.5529449 

### Most accurate is still OOB - Random Forest ###


###-----NZV Dataset-----###

# 70% will be for training; 30% for testing 
# Set seed
set.seed(123)

inTrainingNZV <- createDataPartition(as.factor(iphoneNZV$iphonesentiment), p=0.7, list=FALSE)
dataTrainNZV <- iphoneNZV[inTrainingNZV,]
dataTestNZV <- iphoneNZV[-inTrainingNZV,]

#verify number of observations
nrow(dataTrainNZV) #9083
nrow(dataTestNZV) #3890

################
# Train control
################

# set cross validation -- splits into 10 for cross-validation
fitControl <- trainControl(method="repeatedcv", number=10, repeats=1)

###############
# Train models
###############

### Only using C5.0 and RF, since they performed best with OOB
names(getModelInfo())

# ----- C5.0 Model ----- #
set.seed(123)
C50NZV <- train(iphonesentiment~., data=dataTrainNZV, method="C5.0", importance = T, trControl=fitControl)
C50NZV

#model  winnow  trials  Accuracy   Kappa    
#rules  FALSE    1      0.7563590  0.5205773
#rules  FALSE   10      0.7399570  0.4919145
#rules  FALSE   20      0.7399570  0.4919145
#rules   TRUE    1      0.7565791  0.5208229
#rules   TRUE   10      0.7416073  0.4938208
#rules   TRUE   20      0.7416073  0.4938208
#tree   FALSE    1      0.7551475  0.5191686
#tree   FALSE   10      0.7431470  0.5005746
#tree   FALSE   20      0.7431470  0.5005746
#tree    TRUE    1      0.7556973  0.5203736
#tree    TRUE   10      0.7438061  0.5001276
#tree    TRUE   20      0.7438061  0.5001276

# ----- Random Forest Model ----- #
set.seed(123)
rfNZV <- train(iphonesentiment~., data=dataTrainNZV, method="rf", importance = T, trControl=fitControl)
rfNZV

#mtry  Accuracy   Kappa    
#2    0.7603223  0.5278880
#6    0.7570193  0.5273268
#11    0.7483231  0.5152035

############################
# Predict testSet/validation
############################

# ----- predict with C5.0 Model ----- #
c50PredictNZV <- predict(C50NZV, dataTestNZV)
postResample(c50PredictNZV, dataTestNZV$iphonesentiment)

#Accuracy     Kappa 
#0.7588689 0.5245481 

# ----- predict with Random Forest Model ----- #
rfPredictNZV <- predict(rfNZV, dataTestNZV)
postResample(rfPredictNZV, dataTestNZV$iphonesentiment)

#Accuracy     Kappa 
#0.7586118 0.5218854 

### Most accurate is still OOB - Random Forest ###


###-----RFE Dataset-----###

# 70% will be for training; 30% for testing 
# Set seed
set.seed(123)

inTrainingRFE <- createDataPartition(as.factor(iphoneRFE$iphonesentiment), p=0.7, list=FALSE)
dataTrainRFE <- iphoneRFE[inTrainingRFE,]
dataTestRFE <- iphoneRFE[-inTrainingRFE,]

#verify number of observations
nrow(dataTrainRFE) #9083
nrow(dataTestRFE) #3890

################
# Train control
################

# set cross validation -- splits into 10 for cross-validation
fitControl <- trainControl(method="repeatedcv", number=10, repeats=1)

###############
# Train models
###############

### Only using C5.0 and RF, since they performed best with OOB
names(getModelInfo())

# ----- C5.0 Model ----- #
set.seed(123)
C50RFE <- train(iphonesentiment~., data=dataTrainRFE, method="C5.0", importance = T, trControl=fitControl)
C50RFE

#model  winnow  trials  Accuracy   Kappa    
#rules  FALSE    1      0.7727621  0.5587269
#rules  FALSE   10      0.7597735  0.5397205
#rules  FALSE   20      0.7597735  0.5397205
#rules   TRUE    1      0.7725406  0.5579812
#rules   TRUE   10      0.7601022  0.5404802
#rules   TRUE   20      0.7601022  0.5404802
#tree   FALSE    1      0.7717710  0.5571552
#tree   FALSE   10      0.7607636  0.5420512
#tree   FALSE   20      0.7607636  0.5420512
#tree    TRUE    1      0.7727613  0.5588279
#tree    TRUE   10      0.7620839  0.5437454
#tree    TRUE   20      0.7620839  0.5437454

# ----- Random Forest Model ----- #
set.seed(123)
rfRFE <- train(iphonesentiment~., data=dataTrainRFE, method="rf", importance = T, trControl=fitControl)
rfRFE

#mtry  Accuracy   Kappa    
#2    0.7409402  0.4778095
#10    0.7716614  0.5615740
#18    0.7642856  0.5509646


############################
# Predict testSet/validation
############################

# ----- predict with C5.0 Model ----- #
c50PredictRFE <- predict(C50RFE, dataTestRFE)
postResample(c50PredictRFE, dataTestRFE$iphonesentiment)

#Accuracy     Kappa 
#0.7712082 0.5524619 

# ----- predict with Random Forest Model ----- #
rfPredictRFE <- predict(rfRFE, dataTestRFE)
postResample(rfPredictRFE, dataTestRFE$iphonesentiment)

#Accuracy     Kappa 
#0.7735219 0.5610826


# Since RF OOB and RF RFE are very similar - create confusion matrices
cmRFoob <- confusionMatrix(rfPredictOOB, dataTestOOB$iphonesentiment)
cmRFoob

cmRFrfe <- confusionMatrix(rfPredictRFE, dataTestRFE$iphonesentiment)
cmRFrfe

### Most accurate of all models is RFE - Random Forest ###



############################
# FEATURE ENGINEERING
############################

###########################
### --- Engineer DV --- ###
###########################

# create a new dataset that will be used for recoding sentiment
iphoneRC <- iphoneOOB
# recode sentiment to combine factor levels 0 & 1 and 4 & 5
#iphoneRC$iphonesentiment <- recode(iphoneRC$iphonesentiment, '0' = 1, '1' = 1, '2' = 2, '3' = 3, '4' = 4, '5' = 4) 
iphoneRC$iphonesentiment <- case_when(iphoneRC$iphonesentiment %in% 0 ~ 1,
          iphoneRC$iphonesentiment %in% 1 ~ 1,
          iphoneRC$iphonesentiment %in% 2 ~ 2,
          iphoneRC$iphonesentiment %in% 3 ~ 3,
          iphoneRC$iphonesentiment %in% 4 ~ 4,
          iphoneRC$iphonesentiment %in% 5 ~ 4)
# inspect results
summary(iphoneRC)
str(iphoneRC)
# make iphonesentiment a factor
iphoneRC$iphonesentiment <- as.factor(iphoneRC$iphonesentiment)


### Remodel RF with new dataset

# ----- Split train/test sets ----- #
inTrainingRC <- createDataPartition(as.factor(iphoneRC$iphonesentiment), p=0.7, list=FALSE)
dataTrainRC <- iphoneRC[inTrainingRFE,]
dataTestRC <- iphoneRC[-inTrainingRFE,]

# ----- Random Forest Model ----- #
set.seed(123)
rfRC <- train(iphonesentiment~., data=dataTrainRC, method="rf", importance = T, trControl=fitControl)
rfRC

#mtry  Accuracy   Kappa    
#2    0.7778291  0.3774639
#30    0.8501608  0.6281465
#58    0.8430035  0.6144447

# ----- predict with Random Forest Model ----- #
rfPredictRC <- predict(rfRC, dataTestRC)
postResample(rfPredictRC, dataTestRC$iphonesentiment)

#Accuracy     Kappa 
#0.8491003 0.6231019 

# ----- confusion matrix ----- #
cmRFrc <- confusionMatrix(rfPredictRC, dataTestRC$iphonesentiment)
cmRFrc

plot_ly(iphoneRC, x=~iphoneRC$iphonesentiment, type='histogram')%>%
  layout(title = "Small Matrix - iPhone Sentiment (Predicted)")


###################################
### --- Princ Comp Analysis --- ###
###################################

# data = training and testing from iphoneDF (no feature selection) 
# create object containing centered, scaled PCA components from training set
# excluded the dependent variable and set threshold to .95
preprocessParams <- preProcess(dataTrainOOB[,-59], method=c("center", "scale", "pca"), thresh = 0.95)
print(preprocessParams)

#PCA needed 25 components to capture 95 percent of the variance

preprocessParams2 <- preProcess(dataTrainOOB[,-59], method=c("center", "scale", "pca"), thresh = 0.9)
print(preprocessParams2)

#PCA needed 18 components to capture 90 percent of the variance


# use predict to apply pca parameters, create training, exclude dependant
train.pca <- predict(preprocessParams, dataTrainOOB[,-59])

# add the dependent to training
train.pca$iphonesentiment <- dataTrainOOB$iphonesentiment

# use predict to apply pca parameters, create testing, exclude dependant
test.pca <- predict(preprocessParams, dataTestOOB[,-59])

# add the dependent to training
test.pca$iphonesentiment <- dataTestOOB$iphonesentiment

# inspect results
str(train.pca)
str(test.pca)
nrow(train.pca) #9083
nrow(test.pca) #3890


# ----- Random Forest Model ----- #
set.seed(123)
rf.pca <- train(iphonesentiment~., data=train.pca, method="rf", importance = T, trControl=fitControl)
rf.pca

#mtry  Accuracy   Kappa    
#2    0.7612034  0.5413906
#13    0.7618617  0.5435102
#25    0.7583389  0.5374713

# ----- predict with Random Forest Model ----- #
rfPredict.pca <- predict(rf.pca, test.pca)
postResample(rfPredict.pca, test.pca$iphonesentiment)

#Accuracy     Kappa 
#0.7640103 0.5435245 


# ----- confusion matrix ----- #
cmRF.pca <- confusionMatrix(rfPredict.pca, test.pca$iphonesentiment)
cmRF.pca


#### RF with Recoded Sentiment is the highest accuracy model ####

################################################
# Apply Model to Data - Large Matrix
################################################

iphoneLargeMatrix <- read.csv("iphoneLargeMatrix.csv")

iphonePred <- predict(rfRC, iphoneLargeMatrix)
summary(iphonePred)
str(iphonePred)

#1     2     3     4 
#12082   722   924  6591 

iphoneLargeMatrix$iphonesentiment <- iphonePred

plot_ly(iphoneLargeMatrix, x=~iphoneLargeMatrix$iphonesentiment, type='histogram') %>%
  layout(title = "Large Matrix - iPhone Sentiment (Predicted)")

plot_ly(iphoneLargeMatrix, labels = ~iphonesentiment, type='pie') %>%
  layout(title = "Large Matrix - iPhone Sentiment (Predicted)")


pieDataiphone <- data.frame(COM = c("negative", "somewhat negative", "somewhat positive", "positive"), 
                            values = c(12082, 722, 924, 6591))


# create pie chart
plot_ly(pieDataiphone, labels = ~COM, values = ~ values, type = "pie",
        textposition = 'inside',
        textinfo = 'label+percent',
        insidetextfont = list(color = '#FFFFFF'),
        hoverinfo = 'text',
        text = ~paste( values),
        marker = list(colors = colors,
                      line = list(color = '#FFFFFF', width = 1)),
        showlegend = F) %>%
  layout(title = 'iPhone Sentiment', 
         xaxis = list(showgrid = FALSE, zeroline = FALSE, showticklabels = FALSE),
         yaxis = list(showgrid = FALSE, zeroline = FALSE, showticklabels = FALSE))




#########################
#########################
# Galaxy Data - from POA
#########################
#########################

##################
# Evaluate data
##################

# View first/last obs/rows
head(galaxyOOB)
tail(galaxyOOB)
anyNA(galaxyOOB)
anyDuplicated(galaxyOOB)

colnames(galaxyOOB)
summary(galaxyOOB)
str(galaxyOOB)

# Plot distribution of dependent variable (Galaxy sentiment)
plot_ly(galaxyOOB, x=~galaxyOOB$galaxysentiment, type='histogram')


#######################
# Feature selection
#######################

#######################
# Correlation analysis
#######################

# for regression problems, the below rules apply.
# 1) compare each IV to the DV, if cor > 0.95, remove
# 2) compare each pair of IVs, if cor > 0.90, remove the
#    IV that has the lowest cor to the DV. (see code
#    below for setting a threshold to automatically select
#    IVs that are highly correlated)

# for classification problems, the below rule applies.
# 1) compare each pair of IVs, if cor > 0.90, remove one
#    of the IVs. (see code below to do this programmatically)

options(max.print=1000000)
corrAllgalaxy <- cor(galaxyOOB[,1:58])
corrhighgalaxy <- findCorrelation(corrAllgalaxy, cutoff=0.9) 
corrhighgalaxy
colnames(galaxyOOB[corrhighgalaxy])

#[1] 29 24 56 34 21 31 51 46 16 57 55 30  6  5

#[1] "samsungdisneg" "samsungdispos" "googleperneg" 
#[4] "samsungdisunc" "nokiacamunc"   "nokiadisneg"  
#[7] "nokiaperunc"   "nokiaperneg"   "nokiacamneg"  
#[10] "iosperunc"     "iosperneg"     "sonydisneg"   
#[13] "ios"           "htcphone"     


# plot correlation matrix
corrplot(corrAllgalaxy, method = "circle", main="Correlation of Galaxy OOB Data")
corrplot(corrAllgalaxy, order = "hclust", main="Correlation of Galaxy OOB Data") # sorts based on level of collinearity


# Remove highly correlated variables 
galaxyCOR <- galaxyOOB[,-corrhighgalaxy]

#######################
# Near Zero Variance
#######################

# table to see features with zero variance 
nzvMetricsgalaxy <- nearZeroVar(galaxyOOB, saveMetrics = TRUE)
nzvMetricsgalaxy

# create index to easily remove features 
nzvgalaxy <- nearZeroVar(galaxyOOB, saveMetrics = FALSE)
nzvgalaxy

# create a new data set and remove near zero variance features
galaxyNZV <- galaxyOOB[,-nzvgalaxy]
str(galaxyNZV)

################################
# Recursive Feature Elimination 
################################

# Let's sample the data before using RFE
set.seed(123)
galaxySample <- galaxyOOB[sample(1:nrow(galaxyOOB), 1000, replace=FALSE),]

# Set up rfeControl with randomforest, repeated cross validation and no updates
ctrl2 <- rfeControl(functions = rfFuncs, 
                   method = "repeatedcv",
                   repeats = 5,
                   verbose = FALSE)

# Use rfe and omit the response variable (attribute 59 galaxysentiment) 
rfeResults2 <- rfe(galaxySample[,1:58], 
                  galaxySample$galaxysentiment, 
                  sizes=(1:58), 
                  rfeControl=ctrl2)

# Get results
rfeResults2
#iphone, googleandroid, samsunggalaxy, iphoneperunc, iphoneperpos

# Plot results
plot(rfeResults2, type=c("g", "o"))


# create new data set with rfe recommended features
galaxyRFE <- galaxyOOB[,predictors(rfeResults2)]

# add the dependent variable to iphoneRFE
galaxyRFE$galaxysentiment <- galaxyOOB$galaxysentiment

# review outcome
str(galaxyRFE)

### Four Datasets:
### galaxyOOB
### galaxyCOR
### galaxyNZV
### galaxyRFE

# Convert DV to factor (since it's classification)
galaxyOOB$galaxysentiment <- as.factor(galaxyOOB$galaxysentiment)
galaxyCOR$galaxysentiment <- as.factor(galaxyCOR$galaxysentiment)
galaxyNZV$galaxysentiment <- as.factor(galaxyNZV$galaxysentiment)
galaxyRFE$galaxysentiment <- as.factor(galaxyRFE$galaxysentiment)


##################
# Train/test sets
##################

###-----OOB Dataset-----###

# 70% will be for training; 30% for testing 
# Set seed
set.seed(123)

inTrainingOOBgalaxy <- createDataPartition(as.factor(galaxyOOB$galaxysentiment), p=0.7, list=FALSE)
dataTrainOOBgalaxy <- galaxyOOB[inTrainingOOBgalaxy,]
dataTestOOBgalaxy <- galaxyOOB[-inTrainingOOBgalaxy,]

#verify number of observations
nrow(dataTrainOOBgalaxy) #9040
nrow(dataTestOOBgalaxy) #3871

################
# Train control
################

# set cross validation -- splits into 10 for cross-validation
fitControl <- trainControl(method="repeatedcv", number=10, repeats=1)

###############
# Train models
###############

names(getModelInfo())

# ----- C5.0 Model ----- #
set.seed(123)
C50OOBgalaxy <- train(galaxysentiment~., data=dataTrainOOBgalaxy, method="C5.0", importance = T, trControl=fitControl)
C50OOBgalaxy

#model  winnow  trials  Accuracy   Kappa    
#rules  FALSE    1      0.7653784  0.5294103
#rules  FALSE   10      0.7539856  0.5092183
#rules  FALSE   20      0.7539856  0.5092183
#rules   TRUE    1      0.7658209  0.5298597
#rules   TRUE   10      0.7508891  0.4990666
#rules   TRUE   20      0.7508891  0.4990666
#tree   FALSE    1      0.7628352  0.5249169
#tree   FALSE   10      0.7567509  0.5144032
#tree   FALSE   20      0.7567509  0.5144032
#tree    TRUE    1      0.7636093  0.5258885
#tree    TRUE   10      0.7574152  0.5167121
#tree    TRUE   20      0.7574152  0.5167121

# ----- Random Forest Model ----- #
set.seed(123)
rfOOBgalaxy <- train(galaxysentiment~., data=dataTrainOOBgalaxy, method="rf", importance = T, trControl=fitControl)
rfOOBgalaxy

#mtry  Accuracy   Kappa    
#2    0.7058651  0.3586770
#30    0.7639392  0.5303086
#58    0.7566385  0.5206378


# ----- Support Vector Model ----- #
set.seed(123)
svmOOBgalaxy <- svm(galaxysentiment~., data=dataTrainOOBgalaxy)
svmOOBgalaxy
summary(svmOOBgalaxy)
confusionMatrix(dataTrainOOBgalaxy$galaxysentiment, predict(svmOOBgalaxy))

#Overall Statistics
#Accuracy : 0.7127         
#95% CI : (0.7033, 0.722)
#No Information Rate : 0.849          
#P-Value [Acc > NIR] : 1              
#Kappa : 0.3895  

svmOOB2galaxy <- train(galaxysentiment~., data=dataTrainOOBgalaxy, method="svmLinear2", importance = T, trControl=fitControl)
svmOOB2galaxy

#cost  Accuracy   Kappa    
#0.25  0.6996676  0.3587314
#0.50  0.7063032  0.3796011
#1.00  0.7038711  0.3794882


# ----- Weighted K-Nearest Neighbors Model ----- #
set.seed(123)
kknnOOBgalaxy <- train(galaxysentiment~., data=dataTrainOOBgalaxy, method="kknn", trControl=fitControl, preProcess = c("center", "scale"), tuneLength = 10)
kknnOOBgalaxy

#kmax  Accuracy   Kappa    
#5    0.6590707  0.4098725
#7    0.7342953  0.4899083
#9    0.7204776  0.4779833
#11    0.7421478  0.5014002
#13    0.7467989  0.5051876
#15    0.7511085  0.5086611
#17    0.7513317  0.5083817
#19    0.7522164  0.5100278
#21    0.7483444  0.5059899
#23    0.7558617  0.5126574


############################
# Predict testSet/validation
############################

# ----- predict with C5.0 Model ----- #
c50PredictOOBgalaxy <- predict(C50OOBgalaxy, dataTestOOBgalaxy)
postResample(c50PredictOOBgalaxy, dataTestOOBgalaxy$galaxysentiment)

#Accuracy     Kappa 
#0.7680186 0.5323955 


# ----- predict with Random Forest Model ----- #
rfPredictOOBgalaxy <- predict(rfOOBgalaxy, dataTestOOBgalaxy)
postResample(rfPredictOOBgalaxy, dataTestOOBgalaxy$galaxysentiment)

#Accuracy     Kappa 
#0.7669853 0.5352832 


# ----- predict with Support Vector Model ----- #
svmPredictOOBgalaxy <- predict(svmOOBgalaxy, dataTestOOBgalaxy)
postResample(svmPredictOOBgalaxy, dataTestOOBgalaxy$galaxysentiment)

#Accuracy     Kappa 
#0.7005942 0.3641269 

svmPredictOOB2galaxy <- predict(svmOOB2galaxy, dataTestOOBgalaxy)
postResample(svmPredictOOB2galaxy, dataTestOOBgalaxy$galaxysentiment)

#Accuracy     Kappa 
#0.6974942 0.3642843 


# ----- predict with Weighted K-Nearest Neighbor Model ----- #
kknnPredictOOBgalaxy <- predict(kknnOOBgalaxy, dataTestOOBgalaxy)
postResample(kknnPredictOOBgalaxy, dataTestOOBgalaxy$galaxysentiment)

# Accuracy     Kappa 
#0.7610437 0.5218331 


# Since RF and C5.0 are very similar - create confusion matrices
cmRFgalaxy <- confusionMatrix(rfPredictOOBgalaxy, dataTestOOBgalaxy$galaxysentiment)
cmRFgalaxy

cmc50galaxy <- confusionMatrix(c50PredictOOBgalaxy, dataTestOOBgalaxy$galaxysentiment)
cmc50galaxy

### Most accurate for OOB is C5.0 ###


###-----Correlation Dataset-----###

# 70% will be for training; 30% for testing 
# Set seed
set.seed(123)

inTrainingCORgalaxy <- createDataPartition(as.factor(galaxyCOR$galaxysentiment), p=0.7, list=FALSE)
dataTrainCORgalaxy <- galaxyCOR[inTrainingCORgalaxy,]
dataTestCORgalaxy <- galaxyCOR[-inTrainingCORgalaxy,]

#verify number of observations
nrow(dataTrainCORgalaxy) #9040
nrow(dataTestCORgalaxy) #3933

################
# Train control
################

# set cross validation -- splits into 10 for cross-validation
fitControl <- trainControl(method="repeatedcv", number=10, repeats=1)

###############
# Train models
###############

### Only using C5.0, RF, and KKNN since they performed best with OOB
names(getModelInfo())

# ----- C5.0 Model ----- #
set.seed(123)
C50CORgalaxy <- train(galaxysentiment~., data=dataTrainCORgalaxy, method="C5.0", importance = T, trControl=fitControl)
C50CORgalaxy

#model  winnow  trials  Accuracy   Kappa    
#rules  FALSE    1      0.7652628  0.5292470
#rules  FALSE   10      0.7534269  0.5060976
#rules  FALSE   20      0.7534269  0.5060976
#rules   TRUE    1      0.7645987  0.5281571
#rules   TRUE   10      0.7533131  0.5073356
#rules   TRUE   20      0.7533131  0.5073356
#tree   FALSE    1      0.7648211  0.5292415
#tree   FALSE   10      0.7515479  0.5058213
#tree   FALSE   20      0.7515479  0.5058213
#tree    TRUE    1      0.7639355  0.5275691
#tree    TRUE   10      0.7518737  0.5036482
#tree    TRUE   20      0.7518737  0.5036482

# ----- Random Forest Model ----- #
set.seed(123)
rfCORgalaxy <- train(galaxysentiment~., data=dataTrainCORgalaxy, method="rf", importance = T, trControl=fitControl)
rfCORgalaxy

#mtry  Accuracy   Kappa    
#2    0.6980033  0.3328921
#23    0.7629427  0.5286241
#44    0.7577440  0.5217546

# ----- Weighted K-Nearest Neighbors Model ----- #
set.seed(123)
kknnCORgalaxy <- train(galaxysentiment~., data=dataTrainCORgalaxy, method="kknn", trControl=fitControl, preProcess = c("center", "scale"), tuneLength = 10)
kknnCORgalaxy

#kmax  Accuracy   Kappa    
#5    0.6619526  0.4194660
#7    0.7293154  0.4830374
#9    0.7313085  0.4856255
#11    0.7152645  0.4794489
#13    0.7511027  0.5100222
#15    0.7525406  0.5112493
#17    0.7535381  0.5124030
#19    0.7523223  0.5104371
#21    0.7522088  0.5106426
#23    0.7521021  0.5102825


############################
# Predict testSet/validation
############################

# ----- predict with C5.0 Model ----- #
c50PredictCORgalaxy <- predict(C50CORgalaxy, dataTestCORgalaxy)
postResample(c50PredictCORgalaxy, dataTestCORgalaxy$galaxysentiment)

#Accuracy     Kappa 
#0.7659520 0.5305531 

# ----- predict with Random Forest Model ----- #
rfPredictCORgalaxy <- predict(rfCORgalaxy, dataTestCORgalaxy)
postResample(rfPredictCORgalaxy, dataTestCORgalaxy$galaxysentiment)

#Accuracy     Kappa 
#0.7646603 0.5297197 

# ----- predict with KKNN ----- #
kknnPredictCORgalaxy <- predict(kknnCORgalaxy, dataTestCORgalaxy)
postResample(kknnPredictCORgalaxy, dataTestCORgalaxy$galaxysentiment)

#Accuracy     Kappa 
#0.7607853 0.5213878 

### Most accurate is still OOB - C5.0 ###


###-----NZV Dataset-----###

# 70% will be for training; 30% for testing 
# Set seed
set.seed(123)

inTrainingNZVgalaxy <- createDataPartition(as.factor(galaxyNZV$galaxysentiment), p=0.7, list=FALSE)
dataTrainNZVgalaxy <- galaxyNZV[inTrainingNZVgalaxy,]
dataTestNZVgalaxy <- galaxyNZV[-inTrainingNZVgalaxy,]

#verify number of observations
nrow(dataTrainNZVgalaxy) #9083
nrow(dataTestNZVgalaxy) #3867


################
# Train control
################

# set cross validation -- splits into 10 for cross-validation
fitControl <- trainControl(method="repeatedcv", number=10, repeats=1)

###############
# Train models
###############

### Only using C5.0, RF, and KKNN since they performed best with OOB

# ----- C5.0 Model ----- #
set.seed(123)
C50NZVgalaxy <- train(galaxysentiment~., data=dataTrainNZVgalaxy, method="C5.0", importance = T, trControl=fitControl)
C50NZVgalaxy

#model  winnow  trials  Accuracy   Kappa    
#rules  FALSE    1      0.7543080  0.5023314
#rules  FALSE   10      0.7388230  0.4645818
#rules  FALSE   20      0.7388230  0.4645818
#rules   TRUE    1      0.7539763  0.5019955
#rules   TRUE   10      0.7360596  0.4555703
#rules   TRUE   20      0.7360596  0.4555703
#tree   FALSE    1      0.7543063  0.5034936
#tree   FALSE   10      0.7404812  0.4702989
#tree   FALSE   20      0.7404812  0.4702989
#tree    TRUE    1      0.7540851  0.5031156
#tree    TRUE   10      0.7423632  0.4739152
#tree    TRUE   20      0.7423632  0.4739152

# ----- Random Forest Model ----- #
set.seed(123)
rfNZVgalaxy <- train(galaxysentiment~., data=dataTrainNZVgalaxy, method="rf", importance = T, trControl=fitControl)
rfNZVgalaxy

#mtry  Accuracy   Kappa    
#2    0.7567408  0.5038268
#6    0.7535325  0.5048811
#11    0.7461221  0.4947821

# ----- KKNN Model ----- #
set.seed(123)
kknnNZVgalaxy <- train(galaxysentiment~., data=dataTrainNZVgalaxy, method="kknn", importance = T, trControl=fitControl)
kknnNZVgalaxy

#kmax  Accuracy   Kappa    
#5     0.7021991  0.4438103
#7     0.7193531  0.4609313
#9     0.7319477  0.4743116

############################
# Predict testSet/validation
############################

# ----- predict with C5.0 Model ----- #
c50PredictNZVgalaxy <- predict(C50NZVgalaxy, dataTestNZVgalaxy)
postResample(c50PredictNZVgalaxy, dataTestNZVgalaxy$galaxysentiment)

#Accuracy     Kappa 
#0.7468354 0.4828747 

# ----- predict with Random Forest Model ----- #
rfPredictNZVgalaxy <- predict(rfNZVgalaxy, dataTestNZVgalaxy)
postResample(rfPredictNZVgalaxy, dataTestNZVgalaxy$galaxysentiment)

#Accuracy     Kappa 
#0.7512271 0.4895522

# ----- predict with KKNN Model ----- #
kknnPredictNZVgalaxy <- predict(kknnNZVgalaxy, dataTestNZVgalaxy)
postResample(kknnPredictNZVgalaxy, dataTestNZVgalaxy$galaxysentiment)

#Accuracy     Kappa 
#0.7310772 0.4643420 

### Most accurate is still OOB - C5.0 ###


###-----RFE Dataset-----###

# 70% will be for training; 30% for testing 
# Set seed
set.seed(123)

inTrainingRFEgalaxy <- createDataPartition(as.factor(galaxyRFE$galaxysentiment), p=0.7, list=FALSE)
dataTrainRFEgalaxy <- galaxyRFE[inTrainingRFEgalaxy,]
dataTestRFEgalaxy <- galaxyRFE[-inTrainingRFEgalaxy,]

#verify number of observations
nrow(dataTrainRFEgalaxy) #9040
nrow(dataTestRFEgalaxy) #3871

################
# Train control
################

# set cross validation -- splits into 10 for cross-validation
fitControl <- trainControl(method="repeatedcv", number=10, repeats=1)


###############
# Train models
###############

### Only using C5.0, RF, and KKNN since they performed best with OOB

# ----- C5.0 Model ----- #
set.seed(123)
C50RFEgalaxy <- train(galaxysentiment~., data=dataTrainRFEgalaxy, method="C5.0", importance = T, trControl=fitControl)
C50RFEgalaxy

#model  winnow  trials  Accuracy   Kappa    
#rules  FALSE    1      0.7653778  0.5295603
#rules  FALSE   10      0.7539843  0.5083563
#rules  FALSE   20      0.7539843  0.5083563
#rules   TRUE    1      0.7652671  0.5287745
#rules   TRUE   10      0.7523280  0.5028528
#rules   TRUE   20      0.7523280  0.5028528
#tree   FALSE    1      0.7628352  0.5248674
#tree   FALSE   10      0.7559753  0.5126985
#tree   FALSE   20      0.7559753  0.5126985
#tree    TRUE    1      0.7636095  0.5261409
#tree    TRUE   10      0.7536522  0.5071221
#tree    TRUE   20      0.7536522  0.5071221

# ----- Random Forest Model ----- #
set.seed(123)
rfRFEgalaxy <- train(galaxysentiment~., data=dataTrainRFEgalaxy, method="rf", importance = T, trControl=fitControl)
rfRFEgalaxy

#mtry  Accuracy   Kappa    
#2    0.7112839  0.3741077
#20    0.7639395  0.5306095
#38    0.7560858  0.5196319

# ----- KKNN Model ----- #
set.seed(123)
kknnRFEgalaxy <- train(galaxysentiment~., data=dataTrainRFEgalaxy, method="kknn", importance = T, trControl=fitControl)
kknnRFEgalaxy

#kmax  Accuracy   Kappa    
#5     0.6598458  0.4110526
#7     0.7256730  0.4804402
#9     0.7098483  0.4657305

############################
# Predict testSet/validation
############################

# ----- predict with C5.0 Model ----- #
c50PredictRFEgalaxy <- predict(C50RFEgalaxy, dataTestRFEgalaxy)
postResample(c50PredictRFEgalaxy, dataTestRFEgalaxy$galaxysentiment)

#Accuracy     Kappa 
#0.7672436 0.5317331 

# ----- predict with Random Forest Model ----- #
rfPredictRFEgalaxy <- predict(rfRFEgalaxy, dataTestRFEgalaxy)
postResample(rfPredictRFEgalaxy, dataTestRFEgalaxy$galaxysentiment)

#Accuracy     Kappa 
#0.7667269 0.5349983 

# ----- predict with KKNN Model ----- #
kknnPredictRFEgalaxy <- predict(kknnRFEgalaxy, dataTestRFEgalaxy)
postResample(kknnPredictRFEgalaxy, dataTestRFEgalaxy$galaxysentiment)

#Accuracy     Kappa 
#0.7334022 0.4903109 

### Most accurate is still OOB - C5.0 ###


############################
# FEATURE ENGINEERING
############################

###########################
### --- Engineer DV --- ###
###########################

# create a new dataset that will be used for recoding sentiment
galaxyRC <- galaxyOOB
# recode sentiment to combine factor levels 0 & 1 and 4 & 5
#galaxyRC$galaxysentiment <- recode(galaxyRC$galaxysentiment, '0' = 1, '1' = 1, '2' = 2, '3' = 3, '4' = 4, '5' = 4) 
galaxyRC$galaxysentiment <- case_when(galaxyRC$galaxysentiment %in% 0 ~ 1,
                                      galaxyRC$galaxysentiment %in% 1 ~ 1,
                                      galaxyRC$galaxysentiment %in% 2 ~ 2,
                                      galaxyRC$galaxysentiment %in% 3 ~ 3,
                                      galaxyRC$galaxysentiment %in% 4 ~ 4,
                                      galaxyRC$galaxysentiment %in% 5 ~ 4)
# inspect results
summary(galaxyRC)
str(galaxyRC)
# make galaxysentiment a factor
galaxyRC$galaxysentiment <- as.factor(galaxyRC$galaxysentiment)

### Remodel C5.0 with new dataset

# ----- Split train/test sets ----- #
inTrainingRCgalaxy <- createDataPartition(as.factor(galaxyRC$galaxysentiment), p=0.7, list=FALSE)
dataTrainRCgalaxy <- galaxyRC[inTrainingRCgalaxy,]
dataTestRCgalaxy <- galaxyRC[-inTrainingRCgalaxy,]

# ----- C5.0 Model ----- #
set.seed(123)
C50RCgalaxy <- train(galaxysentiment~., data=dataTrainRCgalaxy, method="C5.0", importance = T, trControl=fitControl)
c50RCgalaxy

#model  winnow  trials  Accuracy   Kappa    
#rules  FALSE    1      0.8417946  0.5883957
#rules  FALSE   10      0.8368187  0.5794156
#rules  FALSE   20      0.8368187  0.5794156
#rules   TRUE    1      0.8413534  0.5877672
#rules   TRUE   10      0.8353805  0.5772947
#rules   TRUE   20      0.8353805  0.5772947
#tree   FALSE    1      0.8411310  0.5875091
#tree   FALSE   10      0.8369256  0.5819116
#tree   FALSE   20      0.8369256  0.5819116
#tree    TRUE    1      0.8410206  0.5875294
#tree    TRUE   10      0.8343868  0.5728799
#tree    TRUE   20      0.8343868  0.5728799

# ----- predict with C5.0 Model ----- #
c50PredictRCgalaxy <- predict(C50RCgalaxy, dataTestRCgalaxy)
postResample(c50PredictRCgalaxy, dataTestRCgalaxy$galaxysentiment)

#Accuracy     Kappa 
#0.8432335 0.5872605 

# ----- confusion matrix ----- #
cmC50rcgalaxy <- confusionMatrix(c50PredictRCgalaxy, dataTestRCgalaxy$galaxysentiment)
cmC50rcgalaxy

plot_ly(galaxyRC, x=~galaxyRC$galaxysentiment, type='histogram')%>%
  layout(title = "Small Matrix - Galaxy Sentiment (Predicted)")


###################################
### --- Princ Comp Analysis --- ###
###################################

# data = training and testing from iphoneDF (no feature selection) 
# create object containing centered, scaled PCA components from training set
# excluded the dependent variable and set threshold to .95
preprocessParamsgalaxy <- preProcess(dataTrainOOBgalaxy[,-59], method=c("center", "scale", "pca"), thresh = 0.95)
print(preprocessParamsgalaxy)

#PCA needed 24 components to capture 95 percent of the variance



# use predict to apply pca parameters, create training, exclude dependant
train.pcagalaxy <- predict(preprocessParamsgalaxy, dataTrainOOBgalaxy[,-59])

# add the dependent to training
train.pcagalaxy$galaxysentiment <- dataTrainOOBgalaxy$galaxysentiment

# use predict to apply pca parameters, create testing, exclude dependant
test.pcagalaxy <- predict(preprocessParamsgalaxy, dataTestOOBgalaxy[,-59])

# add the dependent to training
test.pcagalaxy$galaxysentiment <- dataTestOOBgalaxy$galaxysentiment

# inspect results
str(train.pcagalaxy)
str(test.pcagalaxy)
nrow(train.pcagalaxy) #9040
nrow(test.pcagalaxy) #3871


# ----- C5.0 Model ----- #
set.seed(123)
c50.pcagalaxy <- train(galaxysentiment~., data=train.pcagalaxy,  method="C5.0", importance = T, trControl=fitControl)
c50.pcagalaxy

#model  winnow  trials  Accuracy   Kappa    
#rules  FALSE    1      0.7532086  0.5027494
#rules  FALSE   10      0.7474546  0.4955103
#rules  FALSE   20      0.7474546  0.4955103
#rules   TRUE    1      0.7515479  0.4998539
#rules   TRUE   10      0.7454651  0.4918411
#rules   TRUE   20      0.7454651  0.4918411
#tree   FALSE    1      0.7537616  0.5055266
#tree   FALSE   10      0.7472350  0.4956041
#tree   FALSE   20      0.7472350  0.4956041
#tree    TRUE    1      0.7519911  0.5022218
#tree    TRUE   10      0.7459071  0.4927714
#tree    TRUE   20      0.7459071  0.4927714

# ----- predict with C5.0 Model ----- #
c50Predict.pcagalaxy <- predict(c50.pcagalaxy, test.pcagalaxy)
postResample(c50Predict.pcagalaxy, test.pcagalaxy$galaxysentiment)

#Accuracy     Kappa 
#0.7569104 0.5131477 

# ----- confusion matrix ----- #
cmc50.pcagalaxy <- confusionMatrix(c50Predict.pcagalaxy, test.pcagalaxy$galaxysentiment)
cmc50.pcagalaxy


#### C5.0 with Recoded Sentiment is the highest accuracy model ####


################################################
# Apply Model to Data - Large Matrix
################################################

galaxyLargeMatrix <- read.csv("galaxyLargeMatrix.csv")

galaxyPred <- predict(C50RCgalaxy, galaxyLargeMatrix)
summary(galaxyPred)
str(galaxyPred)

#1     2     3     4 
#11617   710  1174  6818 

galaxyLargeMatrix$galaxysentiment <- galaxyPred

plot_ly(galaxyLargeMatrix, x=~galaxyLargeMatrix$galaxysentiment, type='histogram') %>%
  layout(title = "Large Matrix - Galaxy Sentiment (Predicted)")

plot_ly(galaxyLargeMatrix, labels = ~galaxysentiment, type='pie') %>%
  layout(title = "Large Matrix - Galaxy Sentiment (Predicted)")

pieDatagalaxy <- data.frame(COM = c("negative", "somewhat negative", "somewhat positive", "positive"), 
                     values = c(11617, 710, 1174, 6818))


# create pie chart
plot_ly(pieDatagalaxy, labels = ~COM, values = ~ values, type = "pie",
              textposition = 'inside',
              textinfo = 'label+percent',
              insidetextfont = list(color = '#FFFFFF'),
              hoverinfo = 'text',
              text = ~paste( values),
              marker = list(colors = colors,
                            line = list(color = '#FFFFFF', width = 1)),
              showlegend = F) %>%
  layout(title = 'Galaxy Sentiment', 
         xaxis = list(showgrid = FALSE, zeroline = FALSE, showticklabels = FALSE),
         yaxis = list(showgrid = FALSE, zeroline = FALSE, showticklabels = FALSE))







# Stop Cluster. After performing your tasks, stop your cluster. 
stopCluster(cl)
