library(e1071)
library(tm)
library(data.table)

train = fread("train.csv")
train$id = NULL

#Cleaning of Train Data
corpus = Corpus(VectorSource(train$text))# Get a vector of all the sentences as elements 
corpus = tm_map(corpus, removePunctuation)
corpus = tm_map(corpus, content_transformer(tolower))
corpus = tm_map(corpus, removeNumbers) 
corpus = tm_map(corpus, stripWhitespace)
corpus = tm_map(corpus, removeWords, stopwords('english'))

dtm = DocumentTermMatrix(corpus)
dtm_mat_train = as.matrix(dtm)
freq <- colSums(as.matrix(dtm))

dtm_sm = removeSparseTerms( dtm,0.999)
dtm_mat = as.matrix(dtm_sm)

train_dtm = cbind( train$author, dtm_mat)
train_dtm = as.data.frame(train_dtm)
train_dtm$V1 = as.factor(train_dtm$V1)


#Getting the test data
test = fread("test.csv")
t_id = test$id
test$id = NULL


#Cleaning of Test Data


corpus = Corpus(VectorSource(test$text))# Get a vector of all the sentences as elements
corpus = tm_map(corpus, removePunctuation)
corpus = tm_map(corpus, content_transformer(tolower))
corpus = tm_map(corpus, removeNumbers)
corpus = tm_map(corpus, stripWhitespace)
corpus = tm_map(corpus, removeWords, stopwords('english'))

dtm = DocumentTermMatrix(corpus)
dtm_mat_test = as.matrix(dtm)
freq <- colSums(as.matrix(dtm))

dtm_sm = removeSparseTerms( dtm,0.999)
dtm_mat = as.matrix(dtm_sm)

test_dtm = as.data.frame(dtm_mat)

#test_dtm$V1 = as.factor(test_dtm$V1)

# View(train_dtm)
# View(test_dtm)


train_dtm[2:2498] = lapply(train_dtm[2:2498], function(x) as.numeric(x))

test_dtm[] = lapply(test_dtm[], function(x) as.numeric(x))

test_dtm_smp = test_dtm[1:100, ]
train_dtm_smp = train_dtm[1:100, ]

#SVM implementation


model_svm <- svm(V1 ~ .,data = train_dtm,probability = TRUE)

pred_svm <- predict(model_svm, newdata = test_dtm , probability = TRUE) 

test_pred = cbind(t_id, attr(pred_svm, "probabilities"))
test_pred = as.data.frame(test_pred)

fwrite(test_pred, "svm_submission.csv")

# #Error Calculation
# svm_pred = fread("svm_pred.csv")
# newlab = character(nrow(test_dtm))
# newlab = apply(svm_pred[,2:4], 1, function(x) which.max(x))
# sum(newlab == as.numeric(test_dtm$V1))/length(newlab)
