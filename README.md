# Workers-Compensation-CaseStudy-NLP

1) Built a model to predict high cost workers compnesation claim with almost 100% Testing Accuracy.

2) Built a model to predict whether a claim will go for litigation or not with almost 96% Testing Accuracy

3) Explored through Exploratory Data Analysis and understood the relationship of features.

4) Built Wordcloud to better understand what are the TOP MOST Employee reasons for claiming.

5) Proved that litigation is the key driver of high cost claims using statistical analysis (ANOVA) and graphs.

6) Did alot of Feaure Engineering and built new features to assist the model to better understand the patterns inside the data:
    a) Dervied new features Like Year, Month, Season from Variables including Date (Report, Loss, Closed Date)
    b) Reshaped the entire dataset from 16 columns to 1968 columns with same rows.
    c) Applied Textual Preprocessing and cleaning techniques to create features from CauseDescription Sentences
    d) Using NLP - Bag of Words (Count/TF IDF Vectorizer) Transformed CauseDescription Sentences to Vector Representations
