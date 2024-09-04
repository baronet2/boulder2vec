# Boulder2Vec

This repository contains our code for the paper "Boulder2Vec: Modelling Climber Performances in Professional Bouldering Competitions".

By: Ethan Baron, Victor Hau, Zeke Weng


## Summary

Using data from professional bouldering competitions from 2008 to 2022, we train a logistic regression to predict climber results and measure climber skill. However, this approach is limited, as a single numeric coefficient per climber cannot adequately capture the intricacies of climbers’ varying strengths and weaknesses in different boulder problems. For example, some climbers might prefer more static, technical routes while other climbers may specialize in powerful, dynamic problems.

To this end, we apply Probabilistic Matrix Factorization (PMF), a framework commonly used in recommender systems, to represent the unique characteristics of climbers and problems with latent, multi-dimensional vectors. In this framework, a climber’s performance on a given problem is predicted by taking the dot product of the corresponding climber vector and problem vectors. PMF effectively handles sparse datasets, such as our dataset where only a subset of climbers attempt each particular problem, by extrapolating patterns from similar climbers.

We contrast the empirical performance of PMF to the logistic regression approach and investigate the multivariate representations produced by PMF to gain insights into climber characteristics. Our results show that the multivariate PMF representations improve predictive performance of professional bouldering competitions by capturing both the overall strength of climbers and their specialized skill sets.


## Instructions for Reproduction

To reproduce our results from scratch, follow the steps below.

```
# Install required packages
pip install -r requirements.txt

# Prepare data file
python preprocessing.py

# Fit models
python lr.py
python pmf.py # Warning: this takes a long time to run!

# Scrape climber heights
python scraping_height.py

# Produce performance evaluation results
python eval.py

# Produce embeddings visualizations
python climber_embeddings.py
python problem_embeddings.py

# Produce figures for paper
python eda.py
python figs.py
```
