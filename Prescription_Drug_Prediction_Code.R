# 데이터셋 가져와 범주형 변경 -> 다중 클래스 로지스틱 회귀 -> LASSO 적용 로지스틱 회귀
# -> LDA -> RandomForest -> SVM (Linear/Radial) 순으로 작성 

# ==============================================================================

if (!require("caret")) install.packages("caret", dependencies = TRUE)
if (!require("MASS")) install.packages("MASS", dependencies = TRUE)
library(caret)
library(MASS)
# 데이터 읽기
train_path <- "/Users/kim-yuna/Downloads/MIMIC_DS_F_5000_train.csv"
test_path <- "/Users/kim-yuna/Downloads/MIMIC_DS_F_5000_test.csv"
# 필요한 열의 목록
required_columns <- c("gender", "anchor_age", "BMI", "Blood",
                      "Height", "Weight", "long_title", "pro_long_title", "drug")
# 데이터 읽기: 필요한 열만 선택
train_data <- read.csv(train_path)[, required_columns]
test_data <- read.csv(test_path)[, required_columns]
# 학습 데이터셋 범주형 변환
train_data$long_title <- as.factor(train_data$long_title)
train_data$pro_long_title <- as.factor(train_data$pro_long_title)
train_data$drug <- as.factor(train_data$drug)  # 범주형으로 변환
summary(train_data)
# 테스트 데이터셋 범주형 변환
test_data$long_title <- as.factor(test_data$long_title)
test_data$pro_long_title <- as.factor(test_data$pro_long_title)
test_data$drug <- as.factor(test_data$drug)  # 범주형으로 변환
summary(test_data)



# 다중로지스틱회귀 ==============================================================================
 
# if (!require("VGAM")) install.packages("VGAM", dependencies = TRUE)
library(VGAM)
multinomial_model <- vglm(drug ~ ., family = multinomial, data = train_data)
multinomial_model <- vglm(drug ~ ., family = multinomial, data = train_data)
# 테스트 데이터 예측 및 정확도 계산
multinomial_pred <- predict(multinomial_model, newdata = test_data, type = "response")
multinomial_pred_class <- apply(multinomial_pred, 1, which.max)
accuracy <- mean(multinomial_pred_class == as.numeric(test_data$drug))
cat("Logistic Regression Test Accuracy:", accuracy, "\n")



# LASSO 사용 다중로지스틱 ==============================================================================

# if (!require("glmnet")) install.packages("glmnet", dependencies = TRUE)
library(glmnet)
# 학습 데이터와 테스트 데이터의 범주형 변수 수준 맞추기
train_data$long_title <- factor(train_data$long_title, levels = unique(train_data$long_title))
test_data$long_title <- factor(test_data$long_title, levels = levels(train_data$long_title))
train_data$pro_long_title <- factor(train_data$pro_long_title, levels = unique(train_data$pro_long_title))
test_data$pro_long_title <- factor(test_data$pro_long_title, levels = levels(train_data$pro_long_title))
# 입력 데이터 준비 (model.matrix)
X_train <- model.matrix(drug ~ gender + anchor_age + BMI + Height + Weight +
                          long_title + pro_long_title, data = train_data)[, -1]  
Y_train <- train_data$drug
X_test <- model.matrix(drug ~ gender + anchor_age + BMI + Height + Weight +
                         long_title + pro_long_title, data = test_data)[, -1] 
Y_test <- test_data$drug

# (alpha=1) LASSO 로지스틱 회귀 모델 학습
lasso_model <- glmnet(X_train, Y_train, family = "multinomial", alpha = 1)
# 교차 검증으로 최적의 람다 찾기
set.seed(123)
cv_model <- cv.glmnet(X_train, Y_train, family = "multinomial", alpha = 1)
# 최적의 람다 값
optimal_lambda <- cv_model$lambda.min
cat("Optimal Lambda:", optimal_lambda, "\n")
# 최적의 람다로 예측 수행
lasso_pred <- predict(lasso_model, s = optimal_lambda, newx = X_test, type = "class")
# 정확도 계산
accuracy <- mean(lasso_pred == Y_test)
cat("LASSO Test Accuracy :", accuracy, "\n")



# LDA ==============================================================================

lda.fit = lda(drug~gender+anchor_age+BMI+Height+Blood+
                Weight+long_title+pro_long_title, data = train_data)
lda.fit
output <- capture.output(lda.fit)
cat(output[1:30], sep="\n")
lda.pred=predict(lda.fit, test_data)
lda.class = lda.pred$class
accuracy <- mean(lda.class==test_data$drug)
cat("LDA Test Accuracy:", accuracy, "\n")



# 랜덤 포레스트 ==============================================================================

if (!require("randomForest")) install.packages("randomForest", dependencies = TRUE)
if (!require("caret")) install.packages("caret", dependencies = TRUE)
library(randomForest)
library(caret)

# 범주 수 줄이기: 상위 50개 범주만 남기고 나머지는 "Other"로 그룹화
reduce_levels <- function(column, top_n = 50) {
  # 상위 top_n 범주 계산
  top_levels <- names(sort(table(column), decreasing = TRUE)[1:top_n])
  # 상위 범주에 포함되면 그대로 유지, 아니면 "Other"로 변환
  factor(ifelse(column %in% top_levels, as.character(column), "Other"))
}
# long_title와 pro_long_title 변수 처리
train_data$long_title <- reduce_levels(train_data$long_title, top_n = 50)
train_data$pro_long_title <- reduce_levels(train_data$pro_long_title, top_n = 50)
test_data$long_title <- reduce_levels(test_data$long_title, top_n = 50)
test_data$pro_long_title <- reduce_levels(test_data$pro_long_title, top_n = 50)

# 랜덤 포레스트 모델 학습 (ntree = 500~1000 바꿔가며 진행)
rf_model <- randomForest( drug ~ gender + anchor_age + BMI + Height + Weight + long_title + pro_long_title,
                          data = train_data, ntree = 500, mtry = 3, importance = TRUE)
# 모델 요약 출력
print(rf_model)
# 테스트 데이터로 예측 수행
rf_pred <- predict(rf_model, test_data)
# Confusion Matrix 출력
confusion_matrix <- table(Predicted = rf_pred, Actual = test_data$drug)
# print(confusion_matrix)
# 정확도 계산
accuracy <- mean(rf_pred == test_data$drug)
cat("Random Forest Test Accuracy :", accuracy, "\n")


##SVM ==============================================================================

library(e1071)

#SVM linear==============================================================================

# cost = 0.01, 0.1, 1, 5, 10 바꿔 가며 진행
svm_linear <- svm(x = X_train, y = as.factor(Y_train),, kernel = "linear", cost = 1)
print(svm_linear)
svm_pred_linear <- predict(svm_linear, X_test)
accuracy_linear <- mean(svm_pred_linear == Y_test)
summary(svm_linear)
cat("Test Accuracy (linear):", accuracy_linear, "\n")



# #SVM radial==============================================================================

svm_result_radial <- tune(svm, train.x = X_train, train.y = Y_train, kernel = "radial",
                          ranges = list(cost = c(0.1, 1, 10), gamma = c(0.01, 0.1, 1)))
best_cost_radial <- svm_result_radial$best.parameters$cost
best_gamma_radial <- svm_result_radial$best.parameters$gamma
cat("Best Cost (Radial):", best_cost_radial, "\n")
cat("Best Gamma (Radial):", best_gamma_radial, "\n")
summary(svm_result_radial)

# Radial -> Best cost, gamma
svm_radial <- svm(x = X_train, y = as.factor(Y_train),, kernel = "radial", cost = 1, gamma =1)
print(svm_radial)
svm_pred_radial <- predict(svm_radial, X_test)
accuracy_radial <- mean(svm_pred_radial == Y_test)
summary(svm_radial)
cat("Test Accuracy (Radial):", accuracy_radial, "\n")

