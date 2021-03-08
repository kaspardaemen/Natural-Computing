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
other_language <- read.table('Results/xhosa_3.txt')[,1]
english_res <- read.table('Results/english_3.txt')[,1]

results <- c(english_res, other_language)
print(results)


labels <- c(to_vec(for(i in 1:length(english))  0),  to_vec(for(i in 1:length(other_language))  1))
#r_values <- seq(1,9)
#scores <- read.table(sprintf("Results/hiligaynon_3.txt", r))[,1]
#print(sprintf("Results/hiligaynon_3.txt", r))
#print(length(scores))
roc_df <- get_auc(results, labels, 3)

#for (r in r_values) {
#  scores <- read.table(sprintf("Results/hiligaynon_3.txt", r))[,1]
#  print(sprintf("Results/hiligaynon_3.txt", r))
#  print(length(scores))
#  roc_df <- get_auc(scores, labels, r)
  
#}



    