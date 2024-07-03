# 载入所需的包
library(readxl)
library(pROC)

# 读取Excel文件
file_path <- "D:/桌面/淋巴结转移课题/data.xlsx"
data <- read_excel(file_path)

# logistic回归模型
model <- glm(label ~ ceus + Thickness + t + type + ca199, 
             data = data, family = binomial())

# 输出二元logistic回归分析结果
summary(model)

# 使用模型预测
predictions <- predict(model, type = "response")

# 计算ROC曲线并绘制
roc_curve <- roc(data$label, predictions)
plot(roc_curve, main="ROC Curve", col="red")

# 计算AUC
auc_value <- auc(roc_curve)
print(paste("AUC:", round(auc_value, 4)))

# 计算各参数的OR值及其95%置信区间
# or_values <- exp(coef(model))
# confint_values <- exp(confint(model))
# 
# # 四舍五入，保留4位小数
# or_values_rounded <- round(or_values, 4)
# confint_values_rounded <- round(confint_values, 4)
# 
# # 输出OR值及其95%置信区间
# or_and_confint_rounded <- cbind(OR = or_values_rounded, ConfIntLower = confint_values_rounded[,1], ConfIntUpper = confint_values_rounded[,2])
# print(or_and_confint_rounded)