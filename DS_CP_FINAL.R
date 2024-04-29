# Install and load necessary libraries
library(caret)
library(e1071)
library(ggplot2)
library(ROCR)
library(pROC)
library(dplyr)
library(randomForest)
library(glmnet)

# Load the dataset
lung_cancer_data <- read.csv(file.choose())
cat("Dataset Loaded\n")

# Convert column names to lowercase
colnames(lung_cancer_data) <- tolower(colnames(lung_cancer_data))

# Convert categorical variables to factors
lung_cancer_data$gender <- as.factor(lung_cancer_data$gender)
lung_cancer_data$smoking <- as.factor(lung_cancer_data$smoking)
lung_cancer_data$yellow_fingers <- as.factor(lung_cancer_data$yellow_fingers)
lung_cancer_data$anxiety <- as.factor(lung_cancer_data$anxiety)
lung_cancer_data$peer_pressure <- as.factor(lung_cancer_data$peer_pressure)
lung_cancer_data$chronic_disease <- as.factor(lung_cancer_data$chronic.disease)
lung_cancer_data$fatigue <- as.factor(lung_cancer_data$fatigue)
lung_cancer_data$allergy <- as.factor(lung_cancer_data$allergy)
lung_cancer_data$wheezing <- as.factor(lung_cancer_data$wheezing)
lung_cancer_data$alcohol <- as.factor(lung_cancer_data$alcohol.consuming)
lung_cancer_data$coughing <- as.factor(lung_cancer_data$coughing)
lung_cancer_data$shortness_of_breath <- as.factor(lung_cancer_data$shortness.of.breath)
lung_cancer_data$swallowing_difficulty <- as.factor(lung_cancer_data$swallowing.difficulty)
lung_cancer_data$chest_pain <- as.factor(lung_cancer_data$chest.pain)
lung_cancer_data$lung_cancer <- as.factor(lung_cancer_data$lung_cancer)

# Handle missing data (remove rows with missing values)
lung_cancer_data <- na.omit(lung_cancer_data)
cat("\n")

# Check if there are any missing values left
if (anyNA(lung_cancer_data)) {
  cat("There are still missing values in the dataset.")
} else {
  cat("Missing values have been handled.")
}

# Summary statistics
print(summary(lung_cancer_data))

# Histograms for numeric variables (e.g., Age)
p <- ggplot(lung_cancer_data, aes(x = age)) +
  geom_histogram(binwidth = 5, fill = "blue", color = "black") +
  labs(title = "Age Distribution", x = "Age", y = "Frequency")
print(p)

# Bar plots for Gender
p <- ggplot(lung_cancer_data, aes(x = gender)) +
  geom_bar(fill = "lightgreen") +
  labs(title = "Gender Distribution", x = "Gender", y = "Count")
print(p)

# bar plots for all categorical variables
plot_categorical <- function(data, var_name) {
  ggplot(data, aes(x = {{var_name}})) +
    geom_bar(fill = "lightblue") +
    labs(title = paste(var_name, "Distribution"), x = var_name, y = "Count")
}

# Create a scatter plot of relationship between Age and Lung Cancer
p <- ggplot(lung_cancer_data, aes(x = age, y = lung_cancer, color = lung_cancer)) +
  geom_point() +
  labs(title = "Age vs. Lung Cancer", x = "Age", y = "Lung Cancer", color = "Lung Cancer")
print(p)

# Create a box plot to visualize the distribution of Age by Lung Cancer status
p <- ggplot(lung_cancer_data, aes(x = lung_cancer, y = age, fill = lung_cancer)) +
  geom_boxplot() +
  labs(title = "Distribution of Age by Lung Cancer Status", x = "Lung Cancer", y = "Age", fill = "Lung Cancer")
print(p)

# Create a bar plot to visualize the distribution of Lung Cancer by Gender
p <- ggplot(lung_cancer_data, aes(x = gender, fill = lung_cancer)) +
  geom_bar(position = "fill") +
  labs(title = "Distribution of Lung Cancer by Gender", x = "Gender", y = "Proportion", fill = "Lung Cancer") +
  scale_y_continuous(labels = scales::percent_format(scale = 1))
print(p)

# Perform a chi-squared test of independence between Gender and Lung Cancer
contingency_table <- table(lung_cancer_data$gender, lung_cancer_data$lung_cancer)
chi_square_test <- chisq.test(contingency_table)
print("Chi-Square Test Results for Gender and Lung Cancer:")
print(chi_square_test)

# Split the data into training and testing sets

#set.seed(123)
trainIndex <- createDataPartition(lung_cancer_data$lung_cancer, p = .8, 
                                  list = FALSE, 
                                  times = 1)
data_train <- lung_cancer_data[trainIndex, ]
data_test <- lung_cancer_data[-trainIndex, ]

# Train a classification model (e.g., SVM)
svm_model <- svm(lung_cancer ~ ., data = data_train, kernel = "linear")
svm_predictions <- predict(svm_model, data_test)

# Train a Random Forest classification model   
rf_model <- randomForest(lung_cancer ~ ., data = data_train)
rf_predictions <- predict(rf_model, data_test)

# Train a Logistic Regression model
glm_model <- glm(lung_cancer ~ ., data = data_train, family = binomial)
glm_predictions <- predict(glm_model, data_test, type = "response")

# Ensure levels alignment
glm_predictions <- factor(ifelse(glm_predictions > 0.5, "YES", "NO"), levels = levels(data_test$lung_cancer))


# Convert predictions to factors with the same levels as the test data
levels(glm_predictions) <- levels(data_test$lung_cancer)

# Calculate confusion matrix for SVM
svm_CM <- confusionMatrix(svm_predictions, data_test$lung_cancer)

# Calculate confusion matrix for Random Forest
rf_CM <- confusionMatrix(rf_predictions, data_test$lung_cancer)

# Calculate confusion matrix
glm_CM <- confusionMatrix(glm_predictions, data_test$lung_cancer)



# Calculate metrics for SVM
svm_accuracy <- sum(svm_predictions == data_test$lung_cancer) / length(svm_predictions)
svm_TP <- sum(svm_predictions == "YES" & data_test$lung_cancer == "YES")
svm_FP <- sum(svm_predictions == "YES" & data_test$lung_cancer == "NO")
svm_FN <- sum(svm_predictions == "NO" & data_test$lung_cancer == "YES")
svm_precision <- svm_TP / (svm_TP + svm_FP)
svm_recall <- svm_TP / (svm_TP + svm_FN)
svm_F1 <- 2 * (svm_precision * svm_recall) / (svm_precision + svm_recall)


# Calculate metrics for Random Forest
rf_accuracy <- sum(rf_predictions == data_test$lung_cancer) / length(rf_predictions)
rf_TP <- sum(rf_predictions == "YES" & data_test$lung_cancer == "YES")
rf_FP <- sum(rf_predictions == "YES" & data_test$lung_cancer == "NO")
rf_FN <- sum(rf_predictions == "NO" & data_test$lung_cancer == "YES")
rf_precision <- rf_TP / (rf_TP + rf_FP)
rf_recall <- rf_TP / (rf_TP + rf_FN)
rf_F1 <- 2 * (rf_precision * rf_recall) / (rf_precision + rf_recall)

# Calculate metrics for Logistic Regression
glm_accuracy <- sum(glm_predictions == data_test$lung_cancer) / length(glm_predictions)
glm_TP <- sum(glm_predictions == "YES" & data_test$lung_cancer == "YES")
glm_FP <- sum(glm_predictions == "YES" & data_test$lung_cancer == "NO")
glm_FN <- sum(glm_predictions == "NO" & data_test$lung_cancer == "YES")
glm_precision <- glm_TP / (glm_TP + glm_FP)
glm_recall <- glm_TP / (glm_TP + glm_FN)
glm_F1 <- 2 * (glm_precision * glm_recall) / (glm_precision + glm_recall)

# Calculate sensitivity for all three algorithms
svm_sensitivity <- svm_TP / (svm_TP + svm_FN)
rf_sensitivity <- rf_TP / (rf_TP + rf_FN)
glm_sensitivity <- glm_TP / (glm_TP + glm_FN)



# Create a data frame for metrics comparison
metrics_df <- data.frame(
  Algorithm = c("SVM", "Random Forest", "Logistic Regression"),
  Accuracy = c(svm_accuracy, rf_accuracy, glm_accuracy),
  Precision = c(svm_precision, rf_precision, glm_precision),
  Recall = c(svm_recall, rf_recall, glm_recall),
  Sensitivity=c(svm_sensitivity,rf_sensitivity,glm_sensitivity),
  F1_Score = c(svm_F1, rf_F1, glm_F1))

# Calculate the difference in accuracy from SVM and Random Forest
svm_accuracy_diff <- metrics_df$Accuracy - svm_accuracy
rf_accuracy_diff <- metrics_df$Accuracy - rf_accuracy
glm_accuracy_diff <- metrics_df$Accuracy - glm_accuracy

# Create a data frame for comparison
comparison_df <- data.frame(
  Algorithm = c("SVM", "Random Forest", "Logistic Regression"),
  Accuracy_Difference = c(svm_accuracy_diff, rf_accuracy_diff, glm_accuracy_diff))

# Create a data frame for comparison
comparison_df <- data.frame(
  Algorithm = c("SVM", "Random Forest", "Logistic Regression"),
  Accuracy = c(svm_accuracy, rf_accuracy, glm_accuracy)
)

# Plot the comparison using a bar graph
comparison_plot <- ggplot(comparison_df, aes(x = Algorithm, y = Accuracy, fill = Algorithm)) +
  geom_bar(stat = "identity") +
  labs(title = "Accuracy of Different Algorithms",
       x = "Algorithm", y = "Accuracy") +
  theme_minimal()
# Create a data frame for comparison
comparison_df <- data.frame(
  Algorithm = rep(c("SVM", "Random Forest", "Logistic Regression"), each = nrow(metrics_df)),
  Accuracy = c(svm_accuracy, rf_accuracy, glm_accuracy)
)
# Print the comparison plot
print(comparison_plot)

# Print the metrics table
print(metrics_df)

