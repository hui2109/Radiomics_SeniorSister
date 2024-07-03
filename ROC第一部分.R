library(readxl)
mydata=read.csv('D:/桌面/results.csv')
library(pROC)
m1 <- mydata$Actual_Transition
v1 <- mydata$Predicted_Probability 
roc_1 <- roc(m1, v1)
plot(roc_1,col="blue",legacy.axes=T,)

round(auc(roc_1),3)#求AUC

#delong检验
# roc.test(roc_4, roc_9)

