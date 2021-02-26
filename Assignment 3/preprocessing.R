#install.packages("comprehenr")
library(DescTools)
library(pROC)
library(comprehenr)

chunker <- function(line, chunklength){
  n <- chunklength
  chunks <- substring(line,                     # Apply substring function
                      seq(1, nchar(line), n),
                      seq(n, nchar(line), n))
  return(chunks)
}

preprocess_data <- function (data, chunk_size) {
  data <- list()
  labels <- list()
  for(i in 1:length(data)){
    mylist <- append(data, chunker(data[i]))
  }
}

train <- readLines("syscalls/snd-cert/snd-cert.train")
labels <- readLines("syscalls/snd-cert/snd-cert.1.labels")


