rm(list=ls())
library(rpart)
library(party)
library(tidyverse)
library(caret)
library(doParallel)
library(dplyr)
library(Amelia)
setwd("Z:/Coursera/Practical Machine Learning/Final machine learning")

if (file.exists("pml_training.csv") == FALSE) {
  download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv", "pml_training.csv")
}
if (file.exists("pml_testing.csv") == FALSE) {
  download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv", "pml_testing.csv")
}
col_spec_train <- cols(
  .default = col_double(),
  X1 = col_integer(),
  classe = col_character(),
  user_name = col_character(),
  raw_timestamp_part_1 = col_integer(),
  raw_timestamp_part_2 = col_integer(),
  cvtd_timestamp = col_character(),
  new_window = col_character()
)
col_spec_test <- cols(
  .default = col_double(),
  X1 = col_integer(),
  user_name = col_character(),
  raw_timestamp_part_1 = col_integer(),
  raw_timestamp_part_2 = col_integer(),
  cvtd_timestamp = col_character(),
  new_window = col_character()
)

set.seed(1)

pml_training_raw <- read_csv("pml_training.csv", na = c("#DIV/0!", "", "NA"), trim_ws = TRUE, col_types = col_spec_train)
pml_testing_raw <- read_csv("pml_testing.csv", na = c("#DIV/0!", "", "NA"), trim_ws = TRUE, col_types = col_spec_test)

pml_training <- dplyr::rename(pml_training_raw,
                              user_ = user_name,
                              class = classe,
                              kurtosis_pitch_arm = kurtosis_picth_arm,
                              kurtosis_pitch_belt = kurtosis_picth_belt,
                              kurtosis_pitch_dumbbell = kurtosis_picth_dumbbell,
                              kurtosis_pitch_forearm = kurtosis_picth_forearm) %>% 
  dplyr::mutate(user_ = factor(user_), class = factor(class)) %>%
  cbind(., 
        predict(dummyVars(~ user_, data =.),newdata=.)) %>%
  dplyr::select(class, starts_with("user"), everything(), -starts_with("raw_t"), -cvtd_timestamp, -num_window, -new_window, -user_, -num_window, -X1)

pml_testing <- dplyr::rename(pml_testing_raw,
                             user_ = user_name,
                             kurtosis_pitch_arm = kurtosis_picth_arm,
                             kurtosis_pitch_belt = kurtosis_picth_belt,
                             kurtosis_pitch_dumbbell = kurtosis_picth_dumbbell,
                             kurtosis_pitch_forearm = kurtosis_picth_forearm) %>% 
  dplyr::mutate(user_ = factor(user_)) %>%
  cbind(., 
        predict(dummyVars(~ user_, data =.),newdata=.)) %>%
  dplyr::select(starts_with("user"), everything(), -starts_with("raw_t"), -cvtd_timestamp, -num_window, -new_window, -user_, -num_window, -X1)

missing_perc <- data.frame(train=sort(apply(pml_training, 2, FUN=function(x) mean(is.na(x)))),
                           test=sort(apply(pml_testing, 2, FUN=function(x) mean(is.na(x)))))

pml_training_prune <- pml_training[,rownames(missing_perc)[missing_perc$train<.9]]
pml_testing_prune <- pml_testing[,rownames(missing_perc)[missing_perc$train<.9][-1]]

anyNA(pml_training_prune) | anyNA(pml_training_prune)

prep <- preProcess(pml_training_prune, method="pca")
prep_training <- predict(prep, pml_training_prune) 

# split_row <- filter(pml_training[ceiling(dim(pml_training)[1]*.9):1,], lag(new_window,1)=="yes")$nrow[1]

train_idx <- createDataPartition(prep_training$class, p=.8, list = FALSE)
train_control<- trainControl(method="repeatedcv", number=3, repeats=3, savePredictions = TRUE, allowParallel = TRUE)
tunegrid <- expand.grid(.mtry=c(5)) # 1:10
dtrain <- prep_training[train_idx,]
dtest <- prep_training[-train_idx,]

cores <- detectCores()
cl <- makeCluster(cores)
registerDoParallel(cl)
m_rfo <- train(class~., data = dtrain, metric="Accuracy", method="rf", tuneGrid = tunegrid)
stopCluster(cl)
registerDoSEQ()

m_mul <- train(class~., data = dtrain, method="multinom", verbose=FALSE)
m_gbm <- train(class~., data = dtrain, method="gbm", verbose=FALSE)
m_rnn <- train(class~., data = dtrain, method="nnet", verbose=FALSE)
m_knn <- knn3(class~., data = dtrain)

cm_m_knn <- confusionMatrix(dtest$class,factor(apply(predict(m_knn, dtest), 1, which.max), labels=c("A", "B", "C", "D", "E")))
cm_m_gbm <- confusionMatrix(dtest$class,factor(apply(predict(m_gbm$finalModel, dtest, n.trees = 1), 1, which.max), labels=c("A", "B", "C", "D", "E")))
cm_m_rnn <- confusionMatrix(dtest$class,factor(apply(predict(m_rnn$finalModel, dtest), 1, which.max), labels=c("A", "B", "C", "D", "E")))
cm_m_rfo <- confusionMatrix(dtest$class,predict(m_rfo$finalModel, dtest))
cm_m_mul <- confusionMatrix(dtest$class,predict(m_mul$finalModel, dtest))
model_stats <- t(data_frame(
  knn=cm_m_knn$overall,
  gbm=cm_m_gbm$overall, 
  rnn=cm_m_rnn$overall, 
  rfo=cm_m_rfo$overall, 
  mul=cm_m_mul$overall))
methods <- rownames(model_stats)
colnames(model_stats) <- names(cm_m_gbm$overall)
as_data_frame(model_stats) %>% 
  mutate(m = methods) %>% 
  select(m, everything()) %>% 
  arrange(-Accuracy)
 
preds_test <- data_frame(
  m_mul=predict(m_mul$finalModel, dtest), 
  #m_gbm=factor(apply(predict(m_gbm$finalModel, dtest, n.trees = 1), 1, which.max), labels=c("A", "B", "C", "D", "E")), 
  m_rfo=predict(m_rfo$finalModel, dtest), 
  m_rnn=factor(apply(predict(m_rnn$finalModel, dtest), 1, which.max), labels=c("A", "B", "C", "D", "E")), 
  m_knn=factor(apply(predict(m_knn, dtest), 1, which.max), labels=c("A", "B", "C", "D", "E"))
) %>% mutate(m_rfo2 = m_rfo, m_knn2 = m_knn)
ensemble_preds <- apply(preds_test, 1, FUN=function(r)names(rev(sort(table(r))))[1])
confusionMatrix(dtest$class,factor(ensemble_preds))

# 1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
# A  A  A  A  A  A  A  B  A  A  A  A  B  A  E  A  A  B  B  B 
# F  T  F  T  T  F  F  T  T  T  F  F  T  T  T  F  T  T  T  T
# B  A  C  A  A  E  D  B  A  A  B  C  B  A  E  B  A  B  B  B
# T  T  F  T  T  T  T  T  T  T  T  T  T  T  T  F  T  T  T  T
# B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B


preds_train <- data_frame(
  m_mul=predict(m_mul$finalModel, dtrain), 
  #m_gbm=factor(apply(predict(m_gbm$finalModel, dtrain, n.trees = 1), 1, which.max), labels=c("A", "B", "C", "D", "E")), 
  m_rfo=predict(m_rfo$finalModel, dtrain), 
  m_rnn=factor(apply(predict(m_rnn$finalModel, dtrain), 1, which.max), labels=c("A", "B", "C", "D", "E")), 
  m_knn=factor(apply(predict(m_knn, dtrain), 1, which.max), labels=c("A", "B", "C", "D", "E"))
) 
preds_train$class <- dtrain$class
as_tibble(preds_train)
t1 <- rpart(class~.,preds_train)
t2 <- ctree(class~.,preds_train)
f1 <- predict(t1, preds_test, type="class")
f2 <- predict(t2, preds_test, type="response")

confusionMatrix(dtest$class, f1)
confusionMatrix(dtest$class, f2)

summary(preds)
m_rnn <- train(class~., data = preds, method="rpart", verbose=FALSE)
apply(preds, 1, FUN=function(x)sum(x==x[1])==length(x))
all.equal(c(1,1,1))
?all.equal
ensemble_preds <- apply(preds, 1, FUN=function(r)names(rev(sort(table(r))))[1])
confusionMatrix(dtrain$class,factor(ensemble_preds))