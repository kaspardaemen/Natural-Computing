#install.packages("ggplot2")
library(DescTools)
library(pROC)
library(comprehenr)
library(ggplot2)

get_auc <- function(scores, labels, r) {
  roc_obj <- roc(labels, scores)
  roc_df <- data.frame(
    TPR=rev(roc_obj$sensitivities), 
    FPR=rev(1 - roc_obj$specificities),
    thresholds=rev(roc_obj$thresholds))
  print(auc(roc_obj))
  
  p <- ggplot(roc_df, aes(FPR, TPR)) + 
    geom_path() + labs(title=sprintf("Roc curve of r = %d, the AUC = %f", r, auc(roc_obj)))
  print(p)
  return <- roc_df
  
}

english <- read.table('english.test')[,1]
tagalog <- read.table('tagalog.test')[,1]

labels <- c(to_vec(for(i in 1:length(english))  0),  to_vec(for(i in 1:length(tagalog))  1))
r_values <- seq(1,9)

for (r in r_values) {
  scores <- read.table(sprintf("Results/test_results_%d.txt", r))[,1]
  print(sprintf("Results/test_results_%d.txt", r))
  print(length(scores))
  roc_df <- get_auc(scores, labels, r)
  
}



    