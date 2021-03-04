#install.packages("comprehenr")
library(DescTools)
library(pROC)
library(comprehenr)
library(zeallot)

# setwd("~/Documents/M2/Sem2/Natural_Computing/Assignment3/Repo/Natural-Computing/Assignment 3")
train <- readLines("syscalls/snd-cert/snd-cert.train")

#########################
#preprocess the train set
#########################

chunker <- function(line, chunklength){
  n <- chunklength
  chunks <- substring(line,                     # Apply substring function
                      seq(1, nchar(line), n),
                      seq(n, nchar(line), n))
  return(chunks)
}

chunks <- list()
chunk.labels <- list()
for (i in (1:length(train))){
  line <- train[i]
  c(linechunks, linelabels) <- chunker(line, 7) 
  chunks <- append(chunks, linechunks)
}

vector_chunks <- unlist(chunks)
write(vector_chunks, file = "seven_chunks.txt", ncolumns = 1)


########################
#preprocess the test sets
########################

sets <- c(1:3)
type <- "snd-unm"

chunker <- function(line, chunklength, label){
  n <- chunklength
  chunks <- substring(line,                     # Apply substring function
                      seq(1, nchar(line), n),
                      seq(n, nchar(line), n))
  labels <- to_vec(for(i in 1:length(chunks))  label)
  return(list(chunks, labels))
}


chunks <- list()
chunk.labels <- list()

for (s in (sets)){
  labels <- readLines(sprintf("syscalls/%s/snd-cert.%d.labels", type, s))
  test <- readLines(sprintf("syscalls/%s/snd-cert.%d.test", type, s))
  
  for (i in (1:length(test))){
    line <- test[i]
    label <- labels[i]
    c(linechunks, linelabels) %<-% chunker(line, 7, label ) # used %<-% to unpack a list into two variables
    
    chunks <- append(chunks, linechunks)
    chunk.labels <- append(chunk.labels, linelabels)
  }
  
}

vector_chunks <- unlist(chunks)
vector_chunk_labels <- unlist(chunk.labels)
write(vector_chunks, file = sprintf("merged_%s_test.txt", type), ncolumns = 1)
write(vector_chunk_labels, file = sprintf("merged_%s_labels.txt", type), ncolumns = 1)

