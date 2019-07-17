#library(scatterplot3d)
library(rgl)
library(ggplot2)

dta2 = read.table("SimResult_Crossval.txt",header=T,sep="\t",quote="")
dta = read.table("SimResult_Crossval_full.txt",header=T,sep="\t",quote="")

scatterhistory = 500
minx = 0
maxx = 1500
retainbaseline = T

summary(dta)
summary(dta2)
names(dta)

av1 = cbind(dta$coeff_a,dta$Result)
av2 = cbind(dta$coeff_b,dta$Result)
av3 = cbind(dta$coeff_c,dta$Result)
av4 = cbind(dta$coeff_d,dta$Result)
av5 = cbind(dta$coeff_e,dta$Result)
avnames = c("Coeff A", "Coeff B", "Coeff C", "Coeff D", "Coeff E")
adlist = list(av1,av2,av3,av4,av5)
i = 1
par(mfrow=c(2,2))
for(i in 1:5){
  data = adlist[[i]]
  coeff = data[,1]
  result = data[,2]
  start = length(coeff)-scatterhistory
  if(start<0){start=0}
  end = length(coeff)
  hist(coeff[start:end],main=paste("Distribution of",avnames[i]))
  plot(coeff[start:end],result[start:end],main=paste("Effect on Result",avnames[i]),ylab=avnames[i])
  
}


v1 = cbind(1/dta2$M_Result,1/dta2$SD_Result,1/dta2$Hig_Result,1/dta2$Low_Result)
v2 = cbind(dta2$M_coeff_a,dta2$SD_coeff_a,dta2$Low_coeff_a,dta2$Hig_coeff_a)
v3 = cbind(dta2$M_coeff_b,dta2$SD_coeff_b,dta2$Low_coeff_b,dta2$Hig_coeff_b)
v4 = cbind(dta2$M_coeff_c,dta2$SD_coeff_c,dta2$Low_coeff_c,dta2$Hig_coeff_c)
v5 = cbind(dta2$M_coeff_d,dta2$SD_coeff_d,dta2$Low_coeff_d,dta2$Hig_coeff_d)
v6 = cbind(dta2$M_coeff_e,dta2$SD_coeff_e,dta2$Low_coeff_e,dta2$Hig_coeff_e)

vnames = c("Unexplained Variance", bquote(beta[MediaReliance]), bquote(alpha[MediaImpact]), bquote(alpha[SocialImpact]),bquote(beta[DistanceIdeology]),bquote(beta[Certainty]))
baseline = c(1/2.9440,0,0,0,1,0) #Phase 1
#baseline = c(0.88133576984,0,0,0) #Phase 2
dlist = list(v1,v2,v3,v4,v5,v6)

if(maxx>length(dta2$M_Result)){maxx=length(dta2$M_Result)}
i = 6

par(mfrow=c(2,3))
for(i in c(3,4,5,2,6,1)){
  data = dlist[[i]]
  m = data[,1]
  sd = data[,2]
  low = data[,3]
  high = data[,4]
  min.val = min(low[minx:maxx],na.rm=T)
  if(min.val>baseline[i]&retainbaseline){min.val=baseline[i]}
  max.val = max(high[minx:maxx],na.rm=T)
  if(max.val<baseline[i]&retainbaseline){max.val=baseline[i]}
  lfdn = 1:length(m)
  plot(lfdn,m,type="l",ylim=c(min.val,max.val),xlim=c(minx,maxx),main=vnames[i],ylab="")
  lines(lfdn,high,lty="dashed")
  lines(lfdn,low,lty="dashed")
  abline(baseline[i],0,col="red")
}


b_reli = dta2$M_coeff_a/dta2$M_coeff_b
b_reli = b_reli[minx:length(b_reli)]
mean(b_reli)
sd(b_reli)

b_cert = dta2$M_coeff_e/dta2$M_coeff_c
b_cert = b_cert[minx:length(b_cert)]
mean(b_cert)
sd(b_cert)


real_a = dta2$M_coeff_a*dta2$M_coeff_b
real_a = real_a[minx:length(real_a)]
mean(real_a)
sd(real_a)

real_e = dta2$M_coeff_e*dta2$M_coeff_c
real_e = real_e[minx:length(real_e)]
mean(real_e)
sd(real_e)


dta3 = read.table("Summary_Crossval.txt",header=T,sep="\t",quote="")
dta3$lfdn = 1:length(dta3$TS)
par(mfrow=c(2,3))
## Result
plot(dta3$lfdn,dta3$DRSQ_Training,main="Delta R-Suqared",
     ylim=c(min(c(dta3$DRSQ_Test,dta3$DRSQ_Training)),max(c(dta3$DRSQ_Test,dta3$DRSQ_Training))),
     type="l")
lines(dta3$lfdn,dta3$DRSQ_Test,col="blue")

## Coefficients
plot(dta3$lfdn,dta3$M_coeff_a,type="l",main="A")
plot(dta3$lfdn,dta3$M_coeff_b,type="l",main="B")
plot(dta3$lfdn,dta3$M_coeff_c,type="l",main="C")
plot(dta3$lfdn,dta3$M_coeff_d,type="l",main="D")
plot(dta3$lfdn,dta3$M_coeff_e,type="l",main="E")

par(mfrow=c(1,1))
plot(dta3$DRSQ_Training,dta3$DRSQ_Test)


ggplot(dta3, aes(lfdn)) + 
  geom_line(aes(y = DRSQ_Training, colour = "Training")) + 
  geom_line(aes(y = DRSQ_Test, colour = "Test")) +
  geom_smooth(aes(y=DRSQ_Training, colour = "Training"), method = "lm") +
  geom_smooth(aes(y=DRSQ_Test, colour = "Test"), method = "lm") +
  ylab("Explained Variance") +
  xlab("Bootstrapping Samples") +
  labs(colour="Data set")



### Final Values for all coefficients

ngen = length(dta2$Timestamp)

for(par in c("M_coeff_a","M_coeff_b","M_coeff_c","M_coeff_d","M_coeff_e")){
  clist = dta2[[par]][(ngen-200):ngen]
  print(paste(par,"M:",mean(clist),"; SD:",sd(clist)))
}


