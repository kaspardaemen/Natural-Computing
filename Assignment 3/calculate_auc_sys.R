#install.packages("ggplot2")
library(DescTools)
library(pROC)
library(comprehenr)
library(ggplot2)
library(tidyverse)



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

unm_df <- data.frame(chunk = readLines('merged_snd-unm_test.txt'), label = readLines('merged_snd-unm_labels.txt'))
unm_df$result <- readLines('results-unm.txt')

clean_df <- unm_df %>% filter(nchar(chunk) > 0) %>% mutate(result = as.double(result))

roc_df <- get_auc(clean_df$result, clean_df$label, 3)





    