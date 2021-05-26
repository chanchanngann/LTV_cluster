# Customer LTV forecast with classification models

![](https://github.com/chanchanngann/LTV_cluster/blob/master/image/profit_img.png)

### Background
__*Flaw in identifying profitable customers*__

Marketers commonly estimate customer lifetime value in order to decide which customers are worth continued investment. In the previous project, I identified the best customers using RFM analysis based on historical data. However, I might have overlooked the real option of abandoning unprofitable customer since what I had been using was past figures. That is, the customers that you decided not bother aquiring might be actually profitable in the future.

__*Find the customers who are valuable in the future*__

We might want to identify the future profitable segment for a set of customers using the RFM approach and machine learning models. This helps us pinpoint the potential profitable customers so we could direct marketing dollars to the right customers.

### Objective
To identify the most profitable customer segment in the future among existing customers who have a known transaction history

### Approach

**LTV = revenue - cost^**

1. Period: Define a timeframe for LTV calculation -> Lets say 1 year as the timeframe

    > Since I have 2 years of data available, I will set the middle point as threshold date:
    > - Orders before the threshold date: used to prepare the feature set for model training 
    > - Orders after the threshold date: used to compute the target value (LTV cluster)

2. Feature set: Compute the RFM scores for each customer and use the scores as the feature set 
3. LTV prediction: Use multiclass classfication models to predict 1-year LTV segment for each customer and find the most profitable segment

    > Group into 3 LTV segment: 0 - least profitable, __2 - most profitable__

*^Remark: there is no cost value in the given dataset, assuming cost=0 in this notebook.*
 
### Steps
1. Load & clean the data

2. Split the dataframe into 2 partitions by the threshold date (1st year & 2nd year)

3. Prepare feature set by computing RFM scores using K-means Clustering  (1st year data)

    > - Recency: number of days since most recent purchase date
    > - Frequency: number of transactions within the period
    > - Monetary: total sales attributed to the customer

4. Data preprocessing
5. Build classification models and evaluate model performance
6. Hyperparameter tuning to improve model performance
7. Conclusion

***About the dataset*** 
_This Online Retail II data set contains all the transactions occurring for a UK-based and registered, non-store online retail between 01/12/2009 and 09/12/2011.The company mainly sells unique all-occasion gift-ware. Many customers of the company are wholesalers. Data Source: https://www.kaggle.com/mathchi/online-retail-ii-data-set-from-ml-repository_


***
## Details
### Split the dataframe

The dataset consists of 2 years of transaction data from 2009-12-01 to  2011-12-09. I will take 1 year of data, calculate RFM and use it for predicting next year. 

- Threshold date: 2010-12-01
- 1st year data set: 2009-12-01 to 2010-11-30 (feature set to train the models)
- 2nd year data set: 2010-12-01 to 2011-11-30 (containing target values)

### Caculate the RFM values (1st year data)

Based on the 1st year dataset(df_1styear), I will set the most recent date in the dataset as the observation point to caculate the recency.

- Recency = observation point - most recent transaction date
- Frequency = #invoice counts within the period
- Monetary = Sum of transaction value for each customer (transaction value = Quantity * Price)

![](https://github.com/chanchanngann/LTV_cluster/blob/master/image/01_RFM_distribution.png)

The median values of Recency, Frequency and Monetary are all smaller than Mean values. Their distributions are right-skewed.

### Compute the RFM scores with K-means clustering (1st year data)

The higher scores should indicate better values, i.e. lower recency value should have higher Recency score; higher frequency value should have higher frequency score; higher monetary value should have higher monetary score.

From previous project, the optimal #clusters was 4 which would be set as the parameter for K-means clustering, i.e. scoring will be 0 to 3.

### Total score (1st year data)

Sum up all RFM scores to get an overall score for each customer.

### Segment customers into high, mid, low value based on total score (1st year data)

- high-value: total score = 6 to 8
- mid-value: total score = 3 to 5
- low-value: total score = 0 to 2

![](https://github.com/chanchanngann/LTV_cluster/blob/master/image/02_segment_distribution.png)

High-value segment has lower recency values, higher frequency and monetary values.

### Merge 1st year dataset with 2nd year dataset 

The 2nd year dataset is transformed with only unique customer IDs and monetary values remained. Then we can merge the 1st year with 2nd year dataset by the common customer IDs.

- 1st year data: customer IDs + RFM values + RFM scores + total score + segment
- 2nd year data: customer IDs + Monetary values

![](https://github.com/chanchanngann/LTV_cluster/blob/master/image/03_monetary_2ndyr.png)

### Segment the customers by 2nd year monetary values w/ Kmeans clustering

We will group customers into 3 segments based on monetary values in the 2nd year. Cluster 2 refers to the best customers with highest 2nd year monetary values, while Cluster 0 is the worst with lowest monetary values.

- High LTV --> cluster 2
- Mid LTV --> cluster 1
- Low LTV --> cluster 0

![](https://github.com/chanchanngann/LTV_cluster/blob/master/image/04_LTV_cluster.png)

X-axis refers to the RFM values in the first year, and y-axis indicates the monetary values in the 2nd year with different colors to present the LTV clusters.

- Left: Most of the time high LTV segment has low recency values; while low LTV seg has broad recency pattern.
- Middle: Customers having high money spent in 1st year may or may not spend a lot in 2nd year. Some customers do not spend much in first year turn to spend quite a lot in 2nd year.
- Right: hard to spot out for frequency as most of the points concentrate below 1000.

### Correlation heatmap

![](https://github.com/chanchanngann/LTV_cluster/blob/master/image/05_LTV_correlation.PNG)

Monetary features are mostly correlated with LTV cluster.

### Build classification models

Will apply 3 classifiers and go for the one with best performance:

    A. Support Vector Machine
    B. Random Forest
    C. XGBClassifier
    
### Precision vs Recall in evalutation

Precision = TP/(TP+FP) 

Precision trys to answer: what proportion of positive identifications was actually correct? 
If precision of the model is 50% for cluster 2, when the model predicts the customer is in high LTV segment, it is correct 50% of the time.

Recall = TP/(TP+FN)

Recall trys to answer: what proportion of actual positives was identified correctly?
If recall is 50% for cluster 2, then the model correctly identifies 50% of high LTV customers.

I will use Recall to evaluate the model performance, as it gives a measure of how accurately the model is able to identify the actual high LTV customers. 

![](https://github.com/chanchanngann/LTV_cluster/blob/master/image/06_model_recall.PNG)

The selected models do not perform very well. Let's try to do hyperparameter tuning and see if the result will get better.

### Hyperparameter tuning to improve model performance

We will try to improve the accuracy by tuning the hyperparameters for Random Forest and XGB, with grid search to get the optimized values of hyper parameters. 

The classification report of XGBClassifier:

![](https://github.com/chanchanngann/LTV_cluster/blob/master/image/07_xgb_classification_report.PNG)

From the above result, all of the selected models got 99% recall value when predicting cluster 0. However, for our mostly concerned group - cluster 2 (high LTV value), the models only got 50% in recall value. That means, among the actual high LTV customers, 50% of them could be detected by the models. 


## Conclusion

By forecasting LTV clusters, we are able to pinpoint the potential profitable customers and direct marketing dollars to the right customers.


*Reference:*
- *https://towardsdatascience.com/data-driven-growth-with-python-part-3-customer-lifetime-value-prediction-6017802f2e0f*
- *https://cloud.google.com/architecture/clv-prediction-with-offline-training-intro#:~:text=CLV%20modeling%20can%20help%20you,much%20to%20invest%20in%20advertising.*
- *https://hbr.org/2007/12/the-flaw-in-customer-lifetime-value*
- *https://towardsdatascience.com/churn-prediction-with-machine-learning-c9124d932174*
- *https://developers.google.com/machine-learning/crash-course/classification/precision-and-recall*
- *https://datascience.stackexchange.com/questions/15989/micro-average-vs-macro-average-performance-in-a-multiclass-classification-settin*
- *https://towardsdatascience.com/scale-standardize-or-normalize-with-scikit-learn-6ccc7d176a02*
