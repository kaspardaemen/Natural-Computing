#install.packages("comprehenr")
library(DescTools)
library(pROC)
library(comprehenr)

# setwd("~/Documents/M2/Sem2/Natural_Computing/Assignment3/Repo/Natural-Computing/Assignment 3")
train <- readLines("syscalls/snd-cert/snd-cert.train")
labels <- readLines("syscalls/snd-cert/snd-cert.1.labels")
test <- readLines("syscalls/snd-cert/snd-cert.1.test")

print(head(labels))
print(head(train))

chunker <- function(line, chunklength){
  n <- chunklength
  chunks <- substring(line,                     # Apply substring function
                      seq(1, nchar(line), n),
                      seq(n, nchar(line), n))
  return(chunks)
}

chunks <- list()
for (line in train){
  linechunks <- chunker(line, 7)
  chunks <- append(chunks, linechunks)
}

train <- readLines("syscalls/snd-cert/snd-cert.train")
labels <- readLines("syscalls/snd-cert/snd-cert.1.labels")


write(chunks, file = "data",
      ncolumns = if(is.character(x)) 1 else 5,
      append = FALSE, sep = " ")


