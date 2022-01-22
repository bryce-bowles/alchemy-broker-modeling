library(dplyr)
library(ROCR)
library(caret)
library(rpart)
library(randomForest)
library(cluster)
library(e1071)
library(kernlab)

set.seed(12345)

broker_data <- read.table("alchemy_broker_data.csv", 
                          header=TRUE,
                          colClasses = c(rep("character",3), rep("numeric", 29)),
                          sep=",",
                          row.names=1)

#########################################################################
##Create Model to predict whether GWP will increase or decrease in 2019##
#Adapt the model to predict whether GWP will increase or decrease in 2020##

model_data <- broker_data %>%
  dplyr::select(Submissions_2016, 
                Submissions_2017,
                Submissions_2018,
                QuoteCount_2016,
                QuoteCount_2017,
                QuoteCount_2018,
                GWP_2016,
                GWP_2017,
                GWP_2018, 
                GWP_2019,
                PolicyCount_2016,
                PolicyCount_2017,
                PolicyCount_2018) %>%
  dplyr::mutate(up_no = factor(
    if_else(GWP_2019 > GWP_2018,  "up",  "no",  missing="no"))) %>%
  dplyr::select(-GWP_2019) %>%
  dplyr::rename(GWP1 = GWP_2016, 
                GWP2 = GWP_2017, 
                GWP3 = GWP_2018,
                sub1 = Submissions_2016,
                sub2 = Submissions_2017,
                sub3 = Submissions_2018,
                quo1 = QuoteCount_2016,
                quo2 = QuoteCount_2017,
                quo3 = QuoteCount_2018,
                pol1 = PolicyCount_2016,
                pol2 = PolicyCount_2017,
                pol3 = PolicyCount_2018)
head(model_data)

#if missing then impute 0
model_data[is.na(model_data)] <- 0

#create quote ratios and total policy count
model_data <- model_data %>%
  dplyr::mutate(qr1 = quo1/sub1,
                qr2 = quo2/sub2,
                qr3 = quo3/sub3,
                totpolicy = pol1+pol2+pol3)

head(model_data)
summary(model_data)

# if missing impute 0
model_data$qr1[is.na(model_data$qr1)] <- 0.0
model_data$qr2[is.na(model_data$qr2)] <- 0.0
model_data$qr3[is.na(model_data$qr3)] <- 0.0
#if infinity impute 1
model_data$qr1[is.infinite(model_data$qr1)] <- 1.0
model_data$qr2[is.infinite(model_data$qr2)] <- 1.0
model_data$qr3[is.infinite(model_data$qr3)] <- 1.0

#drop variables that will not be used in model
final_model_data <- model_data %>%
  dplyr::select(GWP1, 
                GWP2,
                GWP3,
                qr1,
                qr2,
                qr3,
                up_no,
                totpolicy)

##################################################################
prediction_data <- broker_data %>%
  dplyr::select(Submissions_2017, 
                Submissions_2018,
                Submissions_2019,
                QuoteCount_2017,
                QuoteCount_2018,
                QuoteCount_2019,
                GWP_2017,
                GWP_2018,
                GWP_2019,
                PolicyCount_2017,
                PolicyCount_2018,
                PolicyCount_2019) %>%
  dplyr::rename(GWP1 = GWP_2017, 
                GWP2 = GWP_2018, 
                GWP3 = GWP_2019,
                sub1 = Submissions_2017,
                sub2 = Submissions_2018,
                sub3 = Submissions_2019,
                quo1 = QuoteCount_2017,
                quo2 = QuoteCount_2018,
                quo3 = QuoteCount_2019, 
                pol1 = PolicyCount_2017,
                pol2 = PolicyCount_2018,
                pol3 = PolicyCount_2019)

#if missing then impute 0
prediction_data[is.na(prediction_data)] <- 0
#create quote ratios and total policy count
prediction_data <- prediction_data %>%
  dplyr::mutate(qr1 = quo1/sub1,
                qr2 = quo2/sub2,
                qr3 = quo3/sub3,
                totpolicy=pol1+pol2+pol3)

head(prediction_data)
summary(prediction_data)

# if missing impute 0
prediction_data$qr1[is.na(prediction_data$qr1)] <- 0.0
prediction_data$qr2[is.na(prediction_data$qr2)] <- 0.0
prediction_data$qr3[is.na(prediction_data$qr3)] <- 0.0
#if infinity impute 1
prediction_data$qr1[is.infinite(prediction_data$qr1)] <- 1.0
prediction_data$qr2[is.infinite(prediction_data$qr2)] <- 1.0
prediction_data$qr3[is.infinite(prediction_data$qr3)] <- 1.0

#drop variables that will not be used in model
final_prediction_data <- prediction_data %>%
  dplyr::select(GWP1, 
                GWP2,
                GWP3,
                qr1,
                qr2,
                qr3,
                totpolicy)



#Partition Data
#75% of data used for training, 142 training obs, 46 test obs
train_rows <- createDataPartition(final_model_data$up_no,
                                  p=0.75,
                                  list=FALSE)
train_broker <- final_model_data[train_rows,]
test_broker <- final_model_data[-train_rows,]
summary(train_broker)
summary(train_broker$up_no)

rpart_broker <- rpart(up_no ~ ., data=train_broker)
rpart_broker_predict <- predict(rpart_broker, test_broker, type="prob")
summary(rpart_broker)
#we get a prob of going up and one for not going up in rpart_broker_predict
rpart_broker_prediction <- prediction(rpart_broker_predict[,2], 
                                      test_broker$up_no,
                                      label.ordering=c("no", "up"))

#get true positive and false positive rate and auc
rpart_broker_performance <- performance(rpart_broker_prediction, "tpr", "fpr")
rpart_broker_auc <- performance(rpart_broker_prediction, "auc")
rpart_broker_auc@y.values[[1]]


############################################################################
##Classification Methods##

#build logistic regression model

my_broker_lr_1 <- glm(up_no ~ ., 
                      data=train_broker, 
                      family=binomial("logit"))
my_broker_lr_predict_1 <- predict(my_broker_lr_1, 
                                  newdata=test_broker, 
                                  type="response")
my_broker_lr_predict_class_1 <- character(length(my_broker_lr_predict_1))
my_broker_lr_predict_class_1[my_broker_lr_predict_1 < 0.5] <- "No"
my_broker_lr_predict_class_1[my_broker_lr_predict_1 >= 0.5] <- "Up"
table(test_broker$up_no, my_broker_lr_predict_class_1)

#misclassification rate
my_broker_lr_cm1<-table(test_broker$up_no, my_broker_lr_predict_class_1)
1-sum(diag(my_broker_lr_cm1))/sum(my_broker_lr_cm1)

#ROC Curve
broker_lr_predict_1 <- predict(my_broker_lr_1, test_broker, type="response")
broker_lr_pred_1 <- prediction(broker_lr_predict_1, 
                               test_broker$up_no,
                               label.ordering=c("no", "up"))
#Variable Importance
varImp(my_broker_lr_1, scale = FALSE)
summary(my_broker_lr_1)

broker_lr_perf_1 <- performance(broker_lr_pred_1, "tpr", "fpr")
broker_lr_auc_1 <- performance(broker_lr_pred_1, "auc")
broker_lr_auc_1@y.values[[1]]


####Classification Tree#####
#build classification tree model
my_broker_rpart_1 <- rpart(up_no ~ ., data=train_broker)

my_broker_rpart_predict_1 <- predict(my_broker_rpart_1, newdata=test_broker, type="class")
table(test_broker$up_no, my_broker_rpart_predict_1)

#variable importance
my_broker_rpart_1$variable.importance

#misclassification rate
my_broker_cm1<-table(test_broker$up_no, my_broker_rpart_predict_1)
1-sum(diag(my_broker_cm1))/sum(my_broker_cm1)

#plot ROC curve
broker_rpart_predict_1  <- predict(my_broker_rpart_1, test_broker, type="prob")
broker_rpart_pred_1 <- prediction(broker_rpart_predict_1[,2], 
                                  test_broker$up_no,
                                  label.ordering=c("no", "up"))
broker_rpart_perf_1 <- performance(broker_rpart_pred_1, "tpr", "fpr")

plot(broker_lr_perf_1, col=1)
plot(broker_rpart_perf_1, col=2, add=TRUE)
legend("bottomright",legend=c("LR","CT"), col=c(1:2),lty=1)

broker_rpart_auc_1 <- performance(broker_rpart_pred_1, "auc")
broker_rpart_auc_1@y.values[[1]]


###########Random Forest##################
broker_rf <- randomForest(up_no ~ ., data = train_broker,
                         importance=TRUE)
broker_rf$importance
broker_predict_rf <- predict(broker_rf, newdata=test_broker, type="class")
(broker_rf_confusion <- table(test_broker$up_no, broker_predict_rf))

1-sum(diag(broker_rf_confusion))/sum(broker_rf_confusion)

my_rf_predict <- predict(broker_rf, newdata=test_broker, type="prob") 
my_rf_pred <- prediction(my_rf_predict[,1],
                                        test_broker$up_no,
                                  label.ordering=c("up", "no")) 
my_rf_perf <- performance(my_rf_pred, "tpr", "fpr")

broker_rf_auc_1 <- performance(my_rf_pred, "auc")
broker_rf_auc_1@y.values[[1]]

plot(my_rf_perf, col=1)

##Variable Performance
broker_rf$importance

######## SVM ############
broker_preprocess <- preProcess(train_broker)
broker_train_num<- predict(broker_preprocess,train_broker)
broker_test_num<- predict(broker_preprocess, test_broker)

#tunes weights
my_broker_svm <- train(up_no ~ ., data=broker_train_num,
                      method="svmLinearWeights",
                      metric="ROC", trControl=trainControl(classProbs=TRUE,
                                                           summaryFunction=twoClassSummary))
my_broker_svm
modelLookup("svmLinearWeights")
my_broker_predict_svm <- predict(my_broker_svm, newdata=broker_test_num) 
(my_broker_svm_confusion <- table(broker_test_num$up_no, my_broker_predict_svm))

my_broker_svm_rbf <- train(up_no ~ ., data=broker_train_num,
                          method="svmRadialWeights", metric="ROC", trControl=trainControl(classProbs=TRUE,
                                                                                summaryFunction=twoClassSummary))
my_broker_predict_svm_rbf <- predict(my_broker_svm_rbf, newdata=broker_test_num) 
(my_broker_svm_rbf_confusion <- table(broker_test_num$up_no, my_broker_predict_svm_rbf))
#there is no (easy) way to assess the importance of predictors in SVM models

my_svm_predict <- predict(my_broker_svm, newdata=broker_test_num, type="prob") 
my_svm_pred <- prediction(my_svm_predict[,1],
                                         broker_test_num$up_no,
                                       label.ordering=c("up", "no")) 
my_svm_perf <- performance(my_svm_pred, "tpr", "fpr")
plot(my_svm_perf, col=2, add=TRUE)

broker_svm_auc_1 <- performance(my_svm_pred, "auc")
broker_svm_auc_1@y.values[[1]]


1-sum(diag(my_broker_svm_rbf_confusion))/sum(my_broker_svm_rbf_confusion)

#####PLOT ROC CURVES
plot(my_rf_perf, col=1)
plot(my_svm_perf, col=2, add=TRUE)
plot(broker_lr_perf_1, col=3, add=TRUE)
plot(broker_rpart_perf_1, col=4, add=TRUE)
legend("bottomright",legend=c("RF","SVM Lin","LR","CT"), col=c(1:4),lty=1)

###############################################################################
#classification trees  model 2020 predictions:
rpart_broker_2020_predict <- predict(rpart_broker, 
                                     final_prediction_data,
                                     type="prob")

#give probs of up and no above
#we want prob of being up
rpart_broker_2020 <- data.frame(broker_id = rownames(broker_data),
                                prediction = rpart_broker_2020_predict[,2])

#gives broker name and prediction(prob of up)
#This is the file we submit below
write.csv(rpart_broker_2020, 
          file="rpart_predictions.csv",
          quote=FALSE, 
          row.names=FALSE)
#############################################################################
#Random Forrest model 2020 Predictions: 
rf_broker_2020_predict <- predict(broker_rf, 
                                   final_prediction_data,
                                   type="prob")

#give probs of up and no above
#we want prob of being up
rf_broker_2020 <- data.frame(broker_id = rownames(broker_data),
                              prediction = rf_broker_2020_predict[,2])

#gives broker name and prediction(prob of up)
#This is the file we submit below
write.csv(rf_broker_2020, 
          file="rf_predictions.csv",
          quote=FALSE)

#########################CLUSTERING####################################
broker_df<-scale(final_prediction_data, center=TRUE, scale=TRUE)
summary(broker_df)

#cluster Dendrogram
broker_dist <- dist(broker_df)
broker_hclust <- hclust(broker_dist)
plot(broker_hclust)

broker_kmeans<-kmeans(broker_df,centers=5)
broker_pcs<-prcomp(broker_df, retx=TRUE)
plot(broker_pcs$x[,1:2], col=broker_kmeans$cluster, pch=broker_kmeans$cluster)
legend("topright", legend=1:5, col=1:5, pch=1:5)
broker_pcs$rotation[,1:3]
broker_kmeans_sil <- silhouette(broker_kmeans$cluster, broker_dist)
plot(broker_kmeans_sil)
summary(broker_pcs)
pca.var <- broker_pcs$sdev^2
pca.var.per <- round(pca.var/sum(pca.var)*100,1)
#Proportion of variables per PC
barplot(pca.var.per, main="Scree Plot", xlab="Principlal Component", ylab="Percent Variation")


#####################################################################

