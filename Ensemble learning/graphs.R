library(comprehenr)
library(ggplot2)

#calculate majority probability with n juries and p competence
mp <- function(n, p){
  k <- floor((n/2)+1)
  return <- (pbinom(k-1, size=n, p=p, lower.tail = FALSE))
}

c <- seq(3, 100, by =2)
p <- 0.6
probability <-to_vec(for(i in c)  mp(i,p))

data <- data.frame(c, probability)

# plot of the number of jury members with a constant competence of 0.6
ggplot(data, aes(x=c, y=probability)) + 
  geom_line() +
  ggtitle('Probability of correct decision for various jury sizes, with a constant competence level p of 0.6') +
  xlab('Jury size (c)') +
  ylab('Probability of the correct decision')
  
  



#plot of multiple competence levels, with a constanct number of predictors/juries of 11
competence <- seq(0.5, 1, by =0.05)
n = 11
probability <-to_vec(for(i in competence)  mp(n,i))

data <- data.frame(competence, probability)

ggplot(data, aes(x=competence, y=probability)) + geom_line() +
  ggtitle('Probability of correct decision for various competence levels, with a constant jury size c of 11') +
  xlab('Competence level (p)') +
  ylab('Probability of the correct decision')





