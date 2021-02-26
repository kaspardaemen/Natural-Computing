#install.packages("comprehenr")
library(DescTools)
library(pROC)
library(comprehenr)

get_auc <- function(scores, labels) {
  roc_obj <- roc(labels, scores)
  roc_df <- data.frame(
    TPR=rev(roc_obj$sensitivities), 
    FPR=rev(1 - roc_obj$specificities),
    thresholds=rev(roc_obj$thresholds))
  print(auc(roc_obj))
  return <- roc_df
}


scores <- read.table('test_results.txt')[,1]
range01 <- function(x){(x-min(x))/(max(x)-min(x))}
english <- read.table('english.test')[,1]
tagalog <- read.table('tagalog.test')[,1]

labels <- c(to_vec(for(i in 1:length(english))  0),  to_vec(for(i in 1:length(tagalog))  1))

roc_df <- get_auc(scores, labels)





    