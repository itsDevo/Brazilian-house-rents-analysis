#Loading necessary libraries

library(gam)
library(xgboost)
library(tidyverse)
library(glmnet)
library(caret)
library(ggplot2)
library(ggrepel)
library(ggpubr)
library(dplyr)
library(plyr)
library(maps)
library(cowplot)
library(MASS)
library(rpart)
library(rpart.plot)
library(gglasso)
library(factoextra)
library(cluster)
library(corrplot)
library(formattable)
library(factoextra)

# Importing the dataset
Data = read.csv("/Users/janvigoje/Desktop/Projec/BrazHousesRent.csv", sep = ",", dec = ".", header = T, colClasses = "character")

head(Data)

#Removing Duplicates
anyDuplicated(Data)
anyNA(Data)

Data <- unique(Data)

anyDuplicated(Data)
anyNA(Data)

#Dealing with Floor value as '-'
count<- sum(Data$floor == "-")
print(paste("Count for floor = '-':", count))

groundFloorHouses <- subset(Data, floor == "-", select = hoa..R..)
groundFloorHouses[,1] = as.numeric(groundFloorHouses[,1])

countHOAzero <- sum(groundFloorHouses[, 1] == 0)

# Checking if '-' means 0 floors
print(paste("Number of observations with floor = '-' and HOA = 0:", countHOAzero))
proportionHOAzero <- countHOAzero / length(groundFloorHouses[, 1])
print(paste("Proportion of HOA = 0:", proportionHOAzero*100,"%"))


#Initializing it as 0
Data[Data == "-"] <- 0

#Categorising variables as numeric, for continous and factor, for categorical.
categorical <- c("city","animal","furniture")
allcols <- colnames(Data)
for(i in 1:ncol(Data)){
  if (allcols[i] %in% categorical){
    Data[,i] = as.factor(Data[,i])
  }else{
    Data[,i] = as.numeric(Data[,i])   #this means they are continuous
  }
}

str(Data)

#Plotting Rent distribution on the map of Brazil
avgrentp <- function(City) {
  cityrent <- subset(Data, city == City, select = rent.amount..R..)
  avgrent <- mean(cityrent$rent.amount..R..)
  return(avgrent)
}

city_data <- data.frame(
  City = c("Belo Horizonte", "Campinas", "Porto Alegre", "Rio de Janeiro", "São Paulo"),
  Latitude = c(-19.9167, -22.9071, -30.0346, -22.9068, -23.5505),
  Longitude = c(-43.9345, -47.0632, -51.2177, -43.1729, -46.6333),
  Average_Rent = sapply(c("Belo Horizonte", "Campinas", "Porto Alegre", "Rio de Janeiro", "São Paulo"), 
                        function(x) avgrentp(x))
)

city_data$Alpha <- (city_data$Average_Rent -
                      min(city_data$Average_Rent)) / (max(city_data$Average_Rent) 
                                                      - min(city_data$Average_Rent))

map <- map_data("world", region = "Brazil")

ggplot() +
  geom_polygon(data = map, aes(x = long, y = lat, group = group), fill = "lightgray", color = "white") +
  geom_label_repel(data = city_data, aes(x = Longitude, y = Latitude, label = City, fill = Average_Rent), color = "black", size = 3,
                   box.padding = 0.5, point.padding = 0.2, force = 1, segment.color = "transparent") +
  geom_point(data = city_data, aes(x = Longitude, y = Latitude), alpha = 0.8, size = 5) +
  labs(title = "Average Rents - Brazil",
       x = "Longitude",
       y = "Latitude") +
  scale_fill_gradient(low = "green", high = "red") +
  theme_minimal()

#Looking at correlation amongst variables
correlations_with_rent <- cor(Data[, sapply(Data, is.numeric)], Data$rent.amount..R..)
correlations_with_rent

#Highest Correlation
cor(Data$fire.insurance..R..,Data$rent.amount..R..)

#Lower Dimensional Models
fit1 <- lm(log(rent.amount..R..) ~ area,data = Data)
summary(fit1)$coefficients

#Making a decision tree
dtree <- rpart(rent.amount..R.. ~ rooms + furniture + bathroom, data = Data, method = "anova")
rpart.plot(dtree)

cont <- c("area","fire.insurance..R..","property.tax..R..","hoa..R..", "floor")

#Removing outliers of continuous variables using IQR
for(i in 1:ncol(Data)){
  if (allcols[i] %in% cont){
    Q3 <- quantile(Data[,i], .75)
    IQR <- IQR(Data[,i])
    Data <- subset(Data, Data[,i]< (Q3 + 3.5*IQR))
  }
}

# Boxplots for discrete variables
discrete <- c("rooms", "floor", "parking.spaces", "bathroom")
for(i in 1:length(discrete)){
 boxplot(Data[, discrete[i]], xlab = discrete[i])
}

#Removing outliers from discrete variables
Data <- subset(Data, floor < 40)
Data <- subset(Data, rooms < 10)
Data <- subset(Data, bathroom < 7)

#Encoding
temp <- Data$animal
temp <- as.character(temp)
temp[temp == "acept"] <- 1      
temp[temp == "not acept"] <- 0       
temp <- as.factor(temp)
Data$animal <- temp

temp <- Data$furniture
temp <- as.character(temp)
temp[temp == "furnished"] <- 1     
temp[temp == "not furnished"] <- 0      
temp <- as.factor(temp)
Data$furniture <- temp

#Making the Traing Set
set.seed(1)
trainrows <- createDataPartition(Data$rent.amount..R.., p=0.8, list=FALSE)
training_set <- Data[trainrows,]
d_test <- Data[-trainrows,]
trainrows <- createDataPartition(training_set$rent.amount..R.., p=0.8, list=FALSE)
d_train <- training_set[trainrows,]
d_val <- training_set[-trainrows,]

unscaled_rent_amounts <- d_test$rent.amount..R.. # We use this later
mean_tr <- mean(training_set$rent.amount..R..)
std_tr <- sd(training_set$rent.amount..R..)

# Scaling
scale_data <- function(dataset, dataset2) {
  for(i in 1:ncol(dataset)){
    if (is.numeric(dataset[1,i])){
      dataset[,i] = scale(dataset[,i], center = mean(dataset2[,i]), scale = sd(dataset2[,i]))
    }
  }
  return(dataset)
} 

# Scaling
d_val <- scale_data(d_val, d_train)
d_test <- scale_data(d_test, training_set)
d_train <- scale_data(d_train, d_train) 
#We scale the training set last since we use its mean and standard deviation to scale the validation set.

d_train_unenc <- d_train
d_val_unenc <- d_val

encode <- function(dataset, excluded=c()) {
  excluded_cols <- dataset[, names(dataset) %in% excluded]
  dataset <- dataset[,!names(dataset) %in% excluded]
  dmy <- dummyVars(" ~ .", data = dataset)
  dataset <- data.frame(predict(dmy, newdata = dataset))
  dataset <- cbind(dataset, excluded_cols)
  return(dataset)
}  

# Encoding the categorical variables in the training, validation and test sets
d_train <- encode(d_train, c("animal", "furniture"))
d_val <- encode(d_val, c("animal", "furniture"))
d_test <- encode(d_test, c("animal", "furniture"))

# Presence of a house in a city would be represented by 0s in the other four city columns, so we can remove one city
d_train <- subset(d_train, select = -c(city.Campinas))
d_val <- subset(d_val, select = -c(city.Campinas))
d_test <- subset(d_test, select = -c(city.Campinas))

# Constudting Models - AIC/BIC (Linear)
full.model <- lm(rent.amount..R.. ~ ., data = d_train)
step.model.aic <- stepAIC(full.model, direction = "both", trace = 0)

step.model.bic <- stepAIC(full.model, direction = "both", trace = 0, k = log1p(nrow(d_train)))

predictions.aic <- predict(step.model.aic, newdata = d_val)
predictions.bic <- predict(step.model.bic, newdata = d_val)


aic_mse <- mean((predictions.aic - d_val$rent.amount..R..)^2)
bic_mse <- mean((predictions.bic - d_val$rent.amount..R..)^2)
aic_rmse <- sqrt(aic_mse)
bic_rmse <- sqrt(bic_mse)
aic_rsquared <- summary(step.model.aic)$r.squared
bic_rsquared <- summary(step.model.bic)$r.squared

table <- data.frame(
  Model = character(),  
  MSE = numeric(),
  RMSE = numeric(),
  R_squared = numeric()
)
table <- rbind(table, c("AIC", round(aic_mse, 5), round(aic_rmse, 5), round(aic_rsquared, 5)))
table <- rbind(table, c("BIC", round(bic_mse, 5), round(bic_rmse, 5), round(bic_rsquared, 5)))
colnames(table) <- c("Model", "MSE", "RMSE", "R_squared")

table

aicplot <- ggplot(data.frame(Fitted = step.model.aic$fitted.values, Residuals = step.model.aic$residuals), aes(x = Fitted, y = Residuals)) +
  geom_point(color = "lightblue") + 
  theme_minimal() +
  geom_hline(yintercept = 0, linetype = "dashed", color = "blue") +
  labs(title = "AIC", x = "Fitted Values", y = "Residuals")
bicplot <- ggplot(data.frame(Fitted = step.model.bic$fitted.values, Residuals = step.model.bic$residuals), aes(x = Fitted, y = Residuals)) +
  geom_point(color = "lightblue") + 
  theme_minimal() +
  geom_hline(yintercept = 0, linetype = "dashed", color = "blue") +
  labs(title = "BIC", x = "Fitted Values", y = "Residuals")
ggarrange(aicplot,bicplot,nrow = 1,ncol = 2)


# Constudting Models - Lasso and Group-Lasso (Penalised Approach)
# Performing 10-fold cross validation
X = model.matrix(rent.amount..R.. ~ ., data=d_train)[,-1]
y = d_train$rent.amount..R..
cvlasso = cv.glmnet(x = X, y = y,nfolds = 10)
groups <- c(1,1,1,1,2,3,4,5,6,7,8,9,10,11)
glasso = cv.gglasso(x=X, y=y, group = groups,nfolds = 10)

# Model Performance
pen_val <- model.matrix(rent.amount..R.. ~ ., data=d_val)[,-1]
lassopredmin <- predict(cvlasso,pen_val, s = "lambda.min")
lmin_mse <- mean((lassopredmin - d_val$rent.amount..R..)^2)
lmin_rmse <- sqrt(lmin_mse)
lmin_Rsq <- cor(lassopredmin,d_val$rent.amount..R..)^2
lassopred1se <- predict(cvlasso,pen_val, s = "lambda.1se")
lse_mse <- mean((lassopred1se - d_val$rent.amount..R..)^2)

glpred1 <- predict(glasso,pen_val, s="lambda.1se")
gl1_MSE <- mean((glpred1 - d_val$rent.amount..R..)^2)
gl1_Rsq <- cor(glpred1,d_val$rent.amount..R..)^2
glpred2 <- predict(glasso,pen_val, s="lambda.min")
gl2_MSE <- mean((glpred2 - d_val$rent.amount..R..)^2)
gl2_Rsq <- cor(glpred2,d_val$rent.amount..R..)^2


table <- data.frame(
  Model = character(),  
  MSE = numeric(),
  RMSE = numeric(),
  R_squared = numeric()
)
table <- rbind(table, c("Lasso", round(lmin_mse, 5), round(sqrt(lmin_mse), 5), round(lmin_Rsq, 5)))
table <- rbind(table, c("grLasso", round(gl2_MSE, 5), round(sqrt(gl2_MSE), 5), round(gl2_Rsq, 5)))
colnames(table) <- c("Model", "MSE", "RMSE", "R_squared")
table

# Constudting Models - GAM (Non-linear Model)
num_names = names(d_train_unenc)[d_train_unenc %>% map_lgl(is.numeric)]
num_names = num_names %>% 
  discard(~.x %in% c("rent.amount..R.."))
num_feat = num_names %>% 
  map_chr(~paste0("s(", .x, ", 10)")) %>%
  paste(collapse = "+")

cat_feat = names(d_train_unenc)[d_train_unenc %>% map_lgl(is.factor)] %>% 
  paste(collapse = "+")

gam_form = as.formula(paste0("rent.amount..R.. ~", num_feat, "+", cat_feat)) # formula
fit_gam = gam(formula = gam_form, family = "gaussian", data = d_train_unenc)

ggplot(data.frame(Fitted = fit_gam$fitted.values, Residuals = fit_gam$residuals), aes(x = Fitted, y = Residuals)) +
  geom_point(color = "lightblue") + 
  theme_minimal() +
  geom_hline(yintercept = 0, linetype = "dashed", color = "red") +
  labs(title = "Residuals vs Fitted", x = "Fitted Values", y = "Residuals")

predicted_values = predict(fit_gam, d_val_unenc) 
observed_values = d_val_unenc$rent.amount..R..
gam_mse = mean((predicted_values - observed_values)^2)
gam_rmse = sqrt(gam_mse)
gam_rsquared = cor(predicted_values, observed_values)^2


# Constudting Models - XGBoost
X_train <- d_train[, !(colnames(d_train) %in% c("rent.amount..R.."))]
y_train <- as.numeric(d_train$rent.amount..R..)

X_val <- d_val[, !(colnames(d_val) %in% c("rent.amount..R.."))]
y_val <- as.numeric(d_val$rent.amount..R..)

for(i in 1:ncol(X_train)){
  X_train[,i] = as.numeric(X_train[,i])
}

for(i in 1:ncol(X_val)){
  X_val[,i] = as.numeric(X_val[,i])
}

xgb_train <- xgb.DMatrix(data = as.matrix(X_train), label = y_train)
xgb_val <- xgb.DMatrix(data = as.matrix(X_val))

#Tunning Parameters
param_grid <- expand.grid(
  nrounds = c(100, 200, 300),
  max_depth = c(3, 4, 5),             
  eta = c(0.01, 0.05, 0.1),
  gamma = 0,
  colsample_bytree = 1,
  min_child_weight = 1,  
  subsample = 1   
)

# Inputting the parameter grid for 10-fold cross-validation to get the optimal combination of values for the parameters
xgb_model <- train(
  X_train, y_train, 
  method = "xgbTree",
  trControl = trainControl(method = "cv", number = 10),
  tuneGrid = param_grid,
  metric = 'RMSE',
  verbosity = 0
)

#NOTE: The other parameters are included and given default values so that they have no influence of the cross-validation (not including them generated an error)

final_model <- xgboost(
  data = xgb_train, 
  nrounds = xgb_model$bestTune$nrounds,
  max_depth = xgb_model$bestTune$max_depth,
  eta = xgb_model$bestTune$eta,
  verbose = 0
)

# evaluating...
predictions <- predict(final_model, newdata = xgb_val)
xgb_mse <- mean((predictions - y_val)^2)
xgb_rmse <- sqrt(xgb_mse)
xgb_rsquared <- 1 - (sum((y_val - predictions)^2) / sum((y_val - mean(y_val))^2))

#Evaluating with scores and Implementing Best Model - XGBoost
table <- data.frame(
  Model = character(),  
  MSE = numeric(),
  RMSE = numeric(),
  R_squared = numeric()
)

table <- rbind(table, c("BIC", round(bic_mse, 5), round(bic_rmse, 5), round(bic_rsquared, 5)))
table <- rbind(table, c("LASSO", round(lmin_mse, 5), round(lmin_rmse, 5), round(lmin_Rsq, 5)))
table <- rbind(table, c("GAM", round(gam_mse, 5), round(gam_rmse, 5), round(gam_rsquared, 5)))
table <- rbind(table, c("XGBoost", round(xgb_mse, 5), round(xgb_rmse, 5), round(xgb_rsquared, 5)))
colnames(table) <- c("Model", "MSE", "RMSE", "R_squared")

table

training_set <- scale_data(training_set, training_set)
training_set <- encode(training_set, c("animal", "furniture"))
training_set <- subset(training_set, select = -c(city.Campinas))

X_train <- training_set[, !(colnames(training_set) %in% c("rent.amount..R.."))]
y_train <- as.numeric(training_set$rent.amount..R..)

X_test <- d_test[, !(colnames(d_test) %in% c("rent.amount..R.."))]
y_test <- as.numeric(d_test$rent.amount..R..)

# Making sure the columns are numeric
for(i in 1:ncol(X_train)){
  X_train[,i] = as.numeric(X_train[,i])
}
for(i in 1:ncol(X_val)){
  X_test[,i] = as.numeric(X_test[,i])
}

xgb_train <- xgb.DMatrix(data = as.matrix(X_train), label = y_train)
xgb_test <- xgb.DMatrix(data = as.matrix(X_test))

final_model <- xgboost(
  data = xgb_train, 
  nrounds = xgb_model$bestTune$nrounds,
  max_depth = xgb_model$bestTune$max_depth,
  eta = xgb_model$bestTune$eta,
  verbose = 0
)

predictions <- predict(final_model, newdata = xgb_test)
unscaled_predictions <- (predictions * std_tr) + mean_tr

# Evaluation of the performance
test_mse <- mean((predictions - y_test)^2)
test_rmse <- sqrt(xgb_mse)
test_rsquared <- 1 - (sum((y_test - predictions)^2) / sum((y_test - mean(y_test))^2))
unscaled_mse <- mean((unscaled_predictions - unscaled_rent_amounts)^2)
unscaled_rmse <- sqrt(unscaled_mse)
print(paste("MSE: ", round(test_mse, 5)))
print(paste("RMSE: ", round(test_rmse, 5), "| RMSE for unscaled predictions", round(unscaled_rmse, 5)))
print(paste("R Squared: ", round(xgb_rsquared, 5)))

prediction_errors <- unscaled_predictions - unscaled_rent_amounts

mean_error <- mean(prediction_errors)
std_error <- sd(prediction_errors)

confidence <- 0.95
z_value <- qnorm(1 - (1 - confidence) / 2)
lower_bound <- mean_error - z_value * std_error 
upper_bound <- mean_error + z_value * std_error 

cat("Confidence Interval (", confidence * 100, "%): [", lower_bound, ", ", upper_bound, "]\n")

importance_scores <- xgb.importance(
  feature_names = colnames(X_train),
  model = final_model
)

importance_data <- as.data.frame(importance_scores)
importance_data$Feature <- factor(importance_data$Feature, levels = importance_data$Feature[order(-importance_data$Gain)])

#Feature Importance
ggplot(importance_data, aes(x = Feature, y = Gain, fill = Gain)) +
  geom_col(show.legend = FALSE) +  
  coord_flip() +  
  scale_fill_gradient(low = "lightblue", high = "darkblue") +  
  labs(
    title = "Feature Importance for Rental Price Prediction",
    x = "Importance",
    y = "Features"
  ) +
  theme_minimal() + 
  theme(
    plot.title = element_text(size = 14, face = "bold", hjust = 0.5),  
    axis.title = element_text(size = 12),  # Axis titles
    axis.text = element_text(size = 10)    # Axis text
  )
#NOTE: The term "gain" is used to indicate how much a particular feature contributes to the model's predictive power, with higher values indicating a more significant contribution.

#Further Exploration on Fire Insurance as a highly correlational variable to Rent Amount
ggplot(Data, aes(x = rent.amount..R.., y = fire.insurance..R..)) +
  geom_point(alpha = 0.6, color = "blue") + 
  labs(
    title = "Rent Amount vs Fire Insurance",
    x = "Rent Amount (R$)",
    y = "Fire Insurance (R$)"
  ) +
  theme_minimal() +  
  theme(
    plot.title = element_text(hjust = 0.5), 
    axis.title.x = element_text(face = "bold", color = "darkblue"),  
    axis.title.y = element_text(face = "bold", color = "darkblue")   
  ) +
  geom_smooth(method = "lm", color = "red", se = FALSE) 


ggplot(Data, aes(x = rent.amount..R.., y = fire.insurance..R.., color = city)) +
  geom_point(alpha = 0.6) +  
  labs(
    title = "Rent Amount vs Fire Insurance",
    x = "Rent Amount (R$)",
    y = "Fire Insurance (R$)"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5),
    axis.title.x = element_text(face = "bold", color = "darkblue"),
    axis.title.y = element_text(face = "bold", color = "darkblue")
  ) +
  geom_smooth(method = "lm", color = "red", se = FALSE) 

hoa_zero_data <- subset(Data,  Data$hoa..R.. == 0 & Data$floor == 0)
hoa_nonzero_data <- subset(Data, Data$hoa..R.. != 0)
hoa_data <- rbind(transform(hoa_zero_data, Group = "Independent"),
                  transform(hoa_nonzero_data, Group = "Condominium"))

ggplot(hoa_data, aes(x = rent.amount..R.., y = fire.insurance..R.., color = Group)) +
  geom_point(alpha = 0.6, size = 3) +  
  labs(
    title = "Rent Amount vs. Fire Insurance Costs",
    x = "Rent Amount (R$)",
    y = "Fire Insurance Cost (R$)"
  ) +
  theme_minimal() +  
  theme(
    plot.title = element_text(hjust = 0.5),  
    plot.subtitle = element_text(hjust = 0.5),  
    axis.title.x = element_text(color = "darkblue"),  
    axis.title.y = element_text(color = "darkblue"),  
    legend.title = element_text(size = 12), 
    legend.text = element_text(size = 10)  
  ) +
  scale_color_manual(values = c("Independent" = "blue", "Condominium" = "green")) +  # Manually setting colors for groups
  geom_smooth(method = "lm", se = FALSE, aes(group = Group), color = "black")  # Adding linear model lines for each group

# removing categoral variables
Data_sc <- Data %>% select_if(is.numeric)
Data_sc <- as.data.frame(scale(Data_sc))

dist_p <- factoextra::get_dist(Data_sc,method = "pearson")
dist_e <- factoextra::get_dist(Data_sc,method = "euclidean")

set.seed(444)

#Choosing K-Means over Hierachial Clustering after thorough consideration
sil_k <- fviz_nbclust(
  Data_sc,
  FUNcluster = kmeans,
  diss = dist_e,
  method = "silhouette",
  print.summary = TRUE,
  k.max = 10
)
sil_h <-fviz_nbclust(
  Data_sc,
  FUNcluster = factoextra::hcut,
  diss = dist_e,
  method = "silhouette",
  print.summary = TRUE,
  k.max = 10
)

#K for each method (acc. to Silhouette)
print(paste("Best k for K-Means:",which(sil_k$data$y == max(sil_k$data$y)),",Best k for for Hierachial Clustering:",which(sil_h$data$y == max(sil_h$data$y))))

#Visualisations
km = kmeans(Data_sc, 2, nstart = 1, iter.max = 1e2)
kmv <- fviz_cluster(km, data = Data_sc, geom = "point", 
                    ggtheme = theme_minimal(), main = "K-Means")
kmv

hcc <- factoextra::hcut(x = dist_e, 
                        k = 4,
                        hc_method = "ward.D2")
ec <-factoextra::fviz_cluster(list(data = Data_sc, cluster = hcc$cluster), main = "Hierarchical",labelsize = 0)

ec 

#Calculating Average Silhouette Width
sk = silhouette(km$cluster, 
                dist = dist_e)
skv <- fviz_silhouette(sk,print.summary = FALSE)
print(paste("K-Means Average Silhouette width:",mean(skv$data$sil_width)))


hcc <- factoextra::hcut(x = dist_e, 
                        k = 4,
                        hc_method = "ward.D2")
print(paste("Hierarchical Clustering Average Silhouette width:",hcc$silinfo$avg.width))

#K-Means is Choosen
#Looking at properties of each cluster
clust1 <- Data[which(km$cluster == 1),]
clust2 <- Data[which(km$cluster == 2),]

cluster_stats <- data.frame(
  Cluster = c("Cluster 1", "Cluster 2"),
  Avg_Rent_Amount = c(round(mean(clust1$rent.amount..R..), 3), round(mean(clust2$rent.amount..R..), 3)),
  Avg_Property_Tax = c(round(mean(clust1$property.tax..R..), 3), round(mean(clust2$property.tax..R..), 3)),
  Avg_Rent_Per_Room = c(round(mean(clust1$rent.amount..R../clust1$rooms), 3), round(mean(clust2$rent.amount..R../clust2$rooms), 3))
)

cluster_stats

#Pie-Chart for each cluser for City
display_city_pie_chart <- function(Data, data_name){  
  
  city_df <- as.data.frame(table(Data$city))
  colnames(city_df) <- c("city", "count")
  
  city_df$percentage <- city_df$count / sum(city_df$count)
  
  ggplot(city_df, aes(x = "", y = percentage, fill = city)) +
    geom_bar(width = 1, stat = "identity") +
    coord_polar("y", start = 0) + 
    scale_fill_brewer(palette = "Set3") +
    theme_void() +
    labs(title = paste("Pie Chart of Cities in", data_name), fill = "City")
}

pie1 <- display_city_pie_chart(clust1,"Cluster1")
pie2 <- display_city_pie_chart(clust2,"Cluster2")
ggarrange(pie1,pie2,nrow = 1,ncol = 2)

