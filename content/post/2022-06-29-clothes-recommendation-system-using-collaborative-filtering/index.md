---
title: |2-

  Clothes Recommendation System Using Collaborative Filtering
author: Will Jarrard
date: '2022-06-29'
slug: []
categories: []
tags: [python, pyspark]
type: ''
subtitle: 'Created a recommendation system in PySpark using ALS modeling to recommend clothes a person is likely to buy next.'
image: ''
---


<!-----

You have some errors, warnings, or alerts. If you are using reckless mode, turn it off to see inline alerts.
* ERRORs: 0
* WARNINGs: 1
* ALERTS: 13

Conversion time: 0.409 seconds.


Using this Markdown file:

1. Paste this output into your source file.
2. See the notes and action items below regarding this conversion run.
3. Check the rendered output (headings, lists, code blocks, tables) for proper
   formatting and use a linkchecker before you publish this page.

Conversion notes:

* Docs to Markdown version 1.0β33
* Wed Jun 29 2022 15:09:45 GMT-0700 (PDT)
* Source doc: big data paper

WARNING:
You have some equations: look for ">>>>>  gd2md-html alert:  equation..." in output.

* Tables are currently converted to HTML tables.
----->

Code can be found [here](https://github.com/wejarrard/h_and_m_recommendations_contest).

## Abstract
Businesses and retailers are constantly faced with the challenge of predicting their customers’ buying behavior. As the realm of machine learning progresses, retailers can predict consumer behavior via recommendation algorithms. Both stakeholders, customers and businesses, benefit from a better prediction algorithm as customers are recommended more of their likes and businesses increase their profit margins. Using an ALS Implicit model, we achieved a mean average precision (MAP) of 0.02. This is a low MAP score but given the incredible amount of articles (over 60,000), it's relatively impressive. Furthermore, our MAP score of 0.02 is close to that of the top performers in the Kaggle competition (~0.03). In the future, we could try more collaborative filtering methods like SVD (singular value decomposition) or some content-based methods. We could also explore neural networks given the vast amount of images for the different articles of clothing.

## Data and Methods
H&M Group released a Kaggle competition called “H&M Personalized Fashion Recommendations” where it wants users to provide recommendations of specific articles to users based on their previous purchases. The dataset consists of a variety of different data, from simple columnar data such as garment type and color, to images of the clothes and text about their descriptions. Four columnar datasets are provided that we decided to narrow down our analysis to: articles.csv, customers.csv, transaction_train.csv. Articles give a detailed description of characteristics of each product. Customers gives basic information of each customer including sex, age group, etc. Transactions_train gives purchases history of customers since September 2018, listed by article_id. 

There are three primary methods that we used to analyze our data. These three methods were association analysis, an implicit ALS model, and an explicit ALS model. The first method, which was our baseline, was to use association analysis on our data. The goal of association rules is to detect relationships or associations between different variables in large data sets. This provided us with a great place to start not only to get a sense of the data we had, but also produce this baseline model that we could measure more complicated models up against. Our goal for this model was to see what items were commonly bought together, and recommend those items to a customer. Given how large our dataset was, we decided to first recommend the type of item based off of other items (for example a swimsuit bottom will recommend a swimsuit top), then recommend specific articles based on age group and gender. The second method we used was an  alternating least squares (ALS) implicit model. ALS modeling is one of the most common collaborative based filtering methods. Its goal is to recommend an item based on other user behaviors. For example, if one user bought these three things, and a second user only bought two of them, it would recommend that third item to the second user. The special thing about ALS implicit modeling is that it is based on data gathered from user behavior, with no ratings from that user needed. This is different from our final model, the ALS explicit model, which needs some sort of rating. In our case, we will use how many times a user bought an item.


These data will need to be preprocessed in two different ways, one for the association analysis, and the other for the two ALS models. The first thing we did for both models was to section out solely the 2020 data. This was still a massive amount of data but the computation power we had was not able to handle the vast amount of data shown in the dataset even when hundreds of gigabytes and tens of cores were requested. For the association analysis, we needed to use table joins on all our different csv files and change variable types before performing an analysis. On the other hand, for our ALS models, we only needed the transaction_train dataset but needed to perform a groupby function on the customer id and article id to get t
he count of how many times a customer bought a particular item. Both of these datasets were split into an 80-20 training test split which will then be fed into the ALS explicit and ALS implicit models.


$$
RMSE = \sqrt{\frac{1}{n}\Sigma_{i=1}^{n}{\Big(\frac{x_i - \hat{x_i}}{\sigma_i}\Big)^2}}
$$

There are two different metrics to compare the results of our models. The first is the root mean squared error (RMSE) of our models (equation 1) where i is the given variable, N is the number of non-missing data points, Xi is the actual observations, and Xi  is the predicted observation. The goal is to minimize this metric. Therefore the closer to 0, the better the model is. This can only be calculated for the ALS implicit and ALS explicit models so a different metric must be used to compare these models to the association analysis. 


$$
MAP@12 = \frac{1}{U}\Sigma_{u=1}^{U}{\Big(\frac{1}{min(m,12)}\Big)} \Sigma_{k=1}^{min(n,12)}{P(k) * rel(k)}
$$

The second metric we will use to evaluate our model is Mean Average Precision (MAP@12) (equation 2) where U is the number of customers, P(k) is the precision at cutoff, n is the number predictions per customer, m is the number of ground truth values per customer, and 𝑟𝑒𝑙(𝑘) is an indicator function equaling 1 if the item at rank k is a relevant (correct) label, zero otherwise. This will be a good way to standardize if our models perform well. 

Results
Training our models utilized 20 cores with over 200 gigabytes requested on UVA’s Rivanna’s Computing Cluster and written in PySpark. The ALS models took great quantities of computing power and time to train while the association analysis was much quicker to train.


|         Model        | RMSE | MAP@12 |
|:--------------------:|:----:|:------:|
| Association Analysis |  N/A | 0.0015 |
|     ALS Implicit     | 1.25 | 0.0221 |
|     ALS Explicit     | 0.49 | 0.0002 |


The ALS implicit performanced the best with a MAP@12 value of 0.0221. This was followed by the association analysis which was a MAP@12 value of 0.0015. Our worst model was the ALS explicit model with a MAP@12 value of 0.0002.

Conclusions
	Overall, our models do not do a great job predicting the next clothing purchase people will make. This is important to the customer as well as the business for both maximizing the customer experience as well as the businesses profit. This task of predicting what a user will buy next is difficult to achieve given the massive amount of different articles that people can buy but ALS performed pretty impressively given the vast challenges mentioned. In the future, more algorithms could be applied including content-based methods such as logistic regression. Another area that could be improved and explored more in depth is more models in collaborative filtering such as singular value decomposition (SVD) or a neural collaborative filtering algorithm. Another area that needs to be looked into is image and text analysis of the photos and descriptions of the clothes. As some people might be more inclined to buy certain types of clothes based on their similarities in appearances or descriptions. This dataset provides many different avenues to pursue and any of these recommendations provide a great starting point for possible extensions to this project to better improve and understand customer’s purchasing behavior.
