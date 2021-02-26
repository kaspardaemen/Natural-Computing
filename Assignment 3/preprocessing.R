#install.packages("comprehenr")
library(DescTools)
library(pROC)
library(comprehenr)


train <- readLines("syscalls/snd-cert/snd-cert.train")
labels <- readLines("syscalls/snd-cert/snd-cert.1.labels")


