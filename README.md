# Alchemy Broker Analysis
Performed segmentation analysis and predictive modeling on insurance broker performance to conclude a random forest model (highest AUC of 73%) predicted whether 2020 Gross Written Premium will increase or decrease from 2019 with a misclassification rate of 35%. Four classification models (classification trees, logistic regression, random forests, and support vector machines) were built, evaluated, and then tuned for prescriptive measures to analyze broker performance. Explored, visualized, and described five groups of brokers using principal component analysis. ([Report](Final_alchemy_broker_project.pdf))

648 Business Data Analytics 901, 11/24/2020

# Alchemy Broker Executive Summary
Alchemy Insurance sought insight into evaluating and predicting broker performance based on historical data. Specifically, the team was tasked with segmentation analysis and predicting whether gross written premium will increase or decrease in the next year.
*	Steps were taken to explore, visualize and describe five groups of brokers using principal component analysis
*	Four predictive models were built, evaluated, and then tuned for prescriptive measures to analyze broker performance 

### Results: 
The top performing cluster was cluster 2. This cluster had higher gross written premiums for the past two years. A random forest model with a high AUC of 0.7321 was used to predict whether the 2020 Gross Written Premium will increase or decrease from 2019 with a misclassification rate of 35%. Important variables for prediction included gross written premium of the past two years and a total of the policy counts of the past three years. The csv file included with this report titled “rf_predictions.csv” contains the probability that the gross written premium will go up in 2020 for each broker id.
### Recommendations:
The random forest model can be used to predict future brokers' performance, identify variables that drive results and impact the company to become more “data driven”. 
 
# Alchemy Broker Segmentation & Performance

### Problem Introduction: 
Alchemy Insurance is seeking ways to manage brokers more effectively and revise incentive compensation plans for the brokers as needed. To do this, Alchemy asked our team to evaluate and predict broker performance based on historical data. Specifically, they asked for a segmentation analysis and predictive model of whether gross written premium will increase or decrease in the next year. 

### Preprocessing:
A data frame was created by selecting Submissions 2016 to 2018, Quote Count 2016 to 2018, Gross Written Premium 2016 to 2019, and Policy Count 2016 to 2018 from the broker dataset. The variables were renamed so that the model could be adapted to predict future years easily. Next, a variable was created called “up no”, which indicated whether the gross written premium had increased from 2018 to 2019. If it has increased it was assigned up, if not then no. Missing values were imputed with a 0. Then quote ratios for each year was calculated as the quote count of that year divided by the submissions. Also, a variable was created summing the policy counts of the three years named total policy. If any of the quote ratios were missing, then a 0 was assigned. If any quote ratios were infinity, then a 1 was assigned. Final variables were selected for the model and classification methods including gross written premium 2016-2018, quote ratios 2016-2018, and total policy. The data was partitioned, 75% training data and 25% test. 
### Broker Segmentation:

The predication data frame was used for clustering. This data frame’s variables included: Gross Written Premium 2017-2019, Quote Ratios 2017-2019 and Total Policy Count of the three years. Summarizing the data after it was centered and scaled showed high max values and indicated that there would be outliers. This led to creating a cluster dendrogram to view hierarchical clustering by calculating pairwise distances between each observation. The graph is gradually right skewed with outliers that are higher than the averages toward the left of the graph. We can see broker 23 and others that are outliers and “far” from the majority of the dataset. 

K-means clustering was then used to create 5 clusters. To visually evaluate the quality of the clusters and how well each point belongs to its cluster, a silhouette plot was used to view the principle component analysis (PCA) scores with an average silhouette width of 0.28. Cluster 2 and cluster 4 have brokers with negative coefficients, indicating poor cluster assignment. This means that there is variation in the data that we cannot see in the PCA plot that overlaps it from other clusters, however the other clusters seem to not have many negative coefficients. 

A summary of the principle component scores was used to calculate a cumulative proportion and bar chart to show that the first two scores explain 77% of the variation in the original dataset.
 
<img width="426" alt="image" src="https://user-images.githubusercontent.com/65502025/152540974-c332c246-26ed-4621-af41-86e068da63f4.png">

 
Cluster 1, Black Circles: This cluster has positive values for PC2 and is centered around 0 for PC1. This indicates these brokers are likely to have high quote ratios in 2018 and 2019.

Cluster 2, Red Triangles: This cluster has positive values for PC1 and is centered around 0 for PC2. This indicates that these brokers are likely to have higher gross written premiums for 2018 and 2019 and higher total policy counts.

Cluster 3, Green Plusses: This cluster has negative values for PC1 and positive values for PC2 which indicates these brokers likely have large quote ratios in 2018 and 2019 and lower total policy count and gross written premiums in 2018 and 2019.

Cluster 4, Blue X’s: This cluster is centered around 0 for both PC1 and PC2. This indicates that these brokers do not have extreme values for the corresponding variables relative to other brokers.

Cluster 5, Aqua Diamonds: This cluster has negative values for PC2 and negative values for PC1. This indicates that these borrowers likely have lower quote ratios for 2018 and 2019, lower gross written premiums in 2018 and 2019 and lower total policy counts.

                         
## Gross Written Premium Prediction:

Different classification methods were tested and analyzed including classification trees, logistic regression, random forests, and support vector machines. The results are shown below.

### Misclassification Rates:
Classification Tree: 0.28
Logistic Regression: 0.35
Random Forests: 0.35
Support Vector Machine: 0.35

### Area Under the Curve:
Classification Tree: 0.7143
Logistic Regression: 0.7242
Random Forests: 0.7321
Support Vector Machine: 0.7341

### ROC Curve:
![image](https://user-images.githubusercontent.com/65502025/151862427-b2c0fcb8-6103-4f1a-9a86-bffccee0a508.png)


### Variable Importance:
Classification Tree:  ![image](https://user-images.githubusercontent.com/65502025/151862453-9779c6b9-59c3-49a4-a2d3-7308ed5e2aad.png)

Gross Written Premium 2018-2019 and total policy count of 2017-2019 are important variables for prediction.

Logistic Regression:

<img width="556" alt="image" src="https://user-images.githubusercontent.com/65502025/152541239-9e1a80a3-5ab0-4dc2-a571-60df6106a7a0.png">
  
Total policy count of 2017-2019 and Quote ratio 2017 are important variables for prediction. 

Random Forests:

<img width="646" alt="image" src="https://user-images.githubusercontent.com/65502025/152541341-7fae8f57-7095-41fd-8fd0-1683da272ed8.png">
 
Total policy count of 2017-2019, gross written premium for 2018 and 2019 and quote ratio for 2017 are important variables for the random forests model. 
Support Vector Machine: There is no easy way to assess the importance of predictors in SVM models.

# Predicting Whether 2020 Gross Written Premium Will Increase:
The random forest model was adapted and applied to the predication data frame. Although the support vector machine model had a slightly higher area under the curve and the classification tree had a lower misclassification rate, the team chose to use a random forest model due to a high AUC while still being able to recognize variable importance. The team felt it was essential for Alchemy to understand the importance of which variables the model used for prediction to know what drives performance. The prediction data frame’s variables included: Gross Written Premium 2017-2019, Quote Ratios 2017-2019 and Total Policy Count of the three years. The csv file included with this report titled “rf_predictions.csv” contains the probability that the gross written premium will go up in 2020 for each broker id. 

# Conclusion:
Alchemy broker performance was evaluated by using historical data to explore, visualize and describe five groups of brokers using principal component analysis. Four different model types were then created to predict broker performance based on historical data. The random forest model performs better than random guessing with a high AUC of 0.7321 and a misclassification rate of 35%. This model can benefit Alchemy to become more “data driven” by predicting brokers' performance in future years while knowing which variables are important to drive broker performance.
 
