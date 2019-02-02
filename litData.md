### Prepare the data

### Case Classifier

We directly went on the lawnet, and use the html file of each cases as our dataset.

We want to know could we just use the contents of the cases to differentiate different type of cases.

We collected 51 trade mark cases and another 50 cases from different categories including ...

From text mining, we able to build a bag of words of 25 which are the most frequent one appears in all the cases. Then by building the tf, i.e., the word frequency vect, we transformed 101 cases to a (101, 25) the matrix.

Then labeled the Trade Mark cases as 1 and the Non-trademarks cases as -1. Finally, used a SVM classifier to train the classifier. 

The results are promissing. The trainset accuracy and testset accuracy are close to 100% after 1000 iterations.

After getting the weights and the bias, in the future the cases can be automatically categoried when uploaded to the lawnet. 

### Win Lose Prediction

After solving the first problem, we found out that in the trade mark cases, there are 6 criterion: marks similarity, visual similarity, ..., and likelyhood of confusion.

After parsing these text from the html files, Lenon manully labeled each section a score. If the score was less than 5, that means the judge think the two trade marks are much more dissimilar than similar. 

So we get 6 similarity scores for 39 trade mark cases. Then we want to find out could these quantified similarity scores be used to predict the win rate of the cases.

So we constructed a dataset of 39 cases, the shape of the cases is (39, 7). The last column of dataset stands for the decision of the cases: win is labeled as 1 and lose is labeled as 0.

This is a binary classification problem, so we trained our model by using logistic regression. In the end the testset accuracy is close to 95%.

So those 6 criterion are not only highly correlated with the case final descision. But also can be used to predict the cases results and give the customer a risk analysis.

### Other researchs

1. There are 3 different types trade mark cases. Text , graph and combined. So we want to use machine learning to differentiate them, which will save the lawyers a lot of time by providing them the most related cases.

   Using similar methods, we built an another bag of words just for the trade mark cases. But this time we ust tf-idf vectorizer which not only takes account the frequncy of each word in a single case but also calculate how many cases has this word in it.

   After that we used naive bayes, but the results are not good enough. One reason is that the limited data we have and based on the number of features we have we need more computational power to support us. 

2. Use native bayes to give the judgs comments on 6 priciples a score, but based on the text frequency analysis, the line between 'similar' and 'disimilar' is vague. We believe if we could gather more data and use NN nets and word embedding, we could definitly get better results in the future.