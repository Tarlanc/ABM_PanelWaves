#library(scatterplot3d)
library(rgl)
library(ggplot2)

dta2 = read.table("SimResult_BS.txt",header=T,sep="\t",quote="")
dta = read.table("SimResult_BS_full.txt",header=T,sep="\t",quote="")

scatterhistory = 500
minx = 1
maxx = 500
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


#v1 = cbind(1/dta2$M_Result,1/dta2$SD_Result,1/dta2$Low_Result,1/dta2$Hig_Result)
v1 = cbind(log(dta2$SD_Result,10),log(dta2$SD_Result,10),log(dta2$SD_Result,10),log(dta2$SD_Result,10))
v2 = cbind(dta2$M_coeff_a,dta2$SD_coeff_a,dta2$Low_coeff_a,dta2$Hig_coeff_a)
v3 = cbind(dta2$M_coeff_b,dta2$SD_coeff_b,dta2$Low_coeff_b,dta2$Hig_coeff_b)
v4 = cbind(dta2$M_coeff_c,dta2$SD_coeff_c,dta2$Low_coeff_c,dta2$Hig_coeff_c)
v5 = cbind(dta2$M_coeff_d,dta2$SD_coeff_d,dta2$Low_coeff_d,dta2$Hig_coeff_d)
v6 = cbind(dta2$M_coeff_e,dta2$SD_coeff_e,dta2$Low_coeff_e,dta2$Hig_coeff_e)

vnames = c(bquote(~"Convergence"), bquote(beta[2]~"(Media Reliance)"), bquote(alpha[1]~"Media Impact"), bquote(alpha[2]~"Social Impact"),bquote(beta[3]~"Relative Distance"),bquote(beta[1]~"Certainty"))
#baseline = c(1/2.9440,0,0,0,1,0) #Phase 1
#baseline = c(0.88133576984,0,0,0) #Phase 2
dlist = list(v1,v2,v3,v4,v5,v6)

baseline = c(1,-.318,.250,.043,1.51,-.003) #Optimal Values for W1

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


gw = 0
bs = 1
dta2$Gen_Within = 0
dta2$Bsample = 0
for(i in 1:length(dta2$Timestamp)){
  gw=gw+1
  if(gw>70){
    gw=1
    bs=bs+1}
  dta2$Gen_Within[i]=gw
  dta2$Bsample[i] = bs
}

good = c()
for(b in 1:bs){
  unexv = dlist[[1]][,1][dta2$Bsample==b]
  minun = min(unexv)
  print(minun)
  if(minun<.34){good=c(good,b)}
}


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
  
  plot(1,1,pch="",ylim=c(min.val,max.val),xlim=c(0,70),main=vnames[i],ylab="",xlab="Generation")
  
  cols = rainbow(bs,alpha=.5)
  
  for(b in good){
    x = c(0,dta2$Gen_Within[dta2$Bsample==b])
    y = c(baseline[i],m[dta2$Bsample==b])
    lines(x,y,lwd=1,col=cols[b])
  }
  #abline(baseline[i],0,col="grey",lty="dashed")
  
}


par(mfrow=c(2,3))
for(par in c("M_coeff_a","M_coeff_b","M_coeff_c","M_coeff_d","M_coeff_e","SD_Result")){
  clist = c()
  for(b in good){
    final = dta2[[par]][dta2$Bsample==b & dta2$Gen_Within==50]
    clist = c(clist,final)
  }
  #print(par)
  #print(clist)
  plot(density(clist),main=par)
  print(paste(par,"M:",mean(clist),"; SD:",sd(clist)))
}
