# MachineLearning_Restaurant-RecommendingSystem

### Overview
This project aims to develop a machine learning-based recommendation system to provide personalized restaurant suggestions. The primary goal is to enhance customer satisfaction by offering tailored restaurant recommendations, drive business engagement, increase the visibility of lesser-known restaurants, and ultimately boost platform engagement. This README details the tools, methodologies, and findings involved in creating the recommender system.

### Team Members
- **202237783** - 이채은
- **202035389** - 조선현
- **202035507** - 김민구
- **201934821** - 유승준

### Business Objectives
1. **Enhance Customer Satisfaction**: Provide personalized restaurant recommendations based on user preferences to improve the dining experience.
2. **Increase Revenue**: Drive visits to specific restaurants to boost overall revenue.
3. **Boost User Engagement**: Increase user interaction frequency with the recommender system, thereby improving engagement metrics.
4. **Increase Visibility**: Raise awareness for lesser-known restaurants by including them in recommendations.

### Data Sources
The data used for this recommender system comes from two main sources:
1. **Kaggle Open Dataset** - A dataset containing restaurant reviews with over 10,000 rows and 8 columns.
2. **Yelp Open Dataset** - A rich dataset comprising millions of reviews, businesses, photos, and user data. This dataset allows for natural language processing (NLP) experiments and is also suitable for recommendation systems.

### Data Exploration & Preprocessing
- **Data Exploration**: Conducted data exploration to understand the underlying features and data relationships. The Yelp dataset included over 6 million reviews, 150,000 businesses, and almost 2 million users, giving a substantial base for building recommendations.
- **Data Preprocessing**: The data preprocessing stage included renaming columns for consistency, extracting only the necessary columns for model training, and ensuring compatibility between datasets. Specifically, the following was done:
  - **Standardization of Column Names**: Adjusted names to maintain consistency between the Kaggle and Yelp datasets.
  - **Top Reviews Extraction**: Extracted the top 10,000 restaurant reviews to help build more effective models.
  - **Feature Engineering**: Extracted adjectives and nouns from reviews to capture customer sentiment and used this information to create new features for the recommendation model.

### Recommendation Techniques
The project utilized **collaborative filtering** for the recommendation system. This approach helps predict the user's preference by evaluating the behavior of similar users. Two main algorithms were applied:

1. **User-Based Collaborative Filtering (KNN-based Approach)**
   - Leveraged the Surprise library for collaborative filtering.
   - Used cosine similarity to measure the similarity between users.
   - Configured parameters such as `k=40` to consider the nearest neighbors and `min_support=5` to ensure reliable predictions.
   - **Results**: Achieved a **Root Mean Square Error (RMSE)** of approximately **1.487**, suggesting good model performance on unseen data.

2. **Matrix Factorization**
   - Created a user-item matrix and filled it with existing ratings.
   - Factorized the matrix to infer missing values, thus predicting users' preferences for unrated items.
   - **Results**: Suggested stores based on matrix completion, with results showing similar predicted ratings due to uniform data distribution.

### Modeling & Evaluation
- **User-Based Filtering**: Implemented user-based collaborative filtering using the Surprise library's **KNNBasic**. The system uses the similarity between users to generate restaurant recommendations.
- **Model Evaluation**: The performance was evaluated using **Root Mean Square Error (RMSE)** to measure prediction accuracy. Key observations include:
  - **Training RMSE**: 0.9301, indicating a good fit.
  - **Test RMSE**: 0.9340, indicating good model generalizability and low overfitting.
  - **Stability**: Minimal difference between training and test RMSE, showing a well-trained model.

### Collaborative Filtering Techniques
Two distinct collaborative filtering models were used:
- **Rating-Based Recommendation** (using **Surprise's KNN**) focuses on numerical ratings given by users. Pros: Straightforward to implement, leverages structured data effectively. Cons: Lacks the ability to understand the detailed context behind each rating.
- **Text-Based Recommendation** (using KNN with **TF-IDF**): Combines both ratings and text-based features to create recommendations. This approach leverages richer data features derived from review texts. Pros: Provides more contextual insights. Cons: Requires substantial computational resources and more extensive preprocessing.

### Results and Limitations
- **Precision@K and Recall@K**: Achieved 0.0 for one random user, largely due to dataset limitations with no explicit relevance labels.
- **Mean Average Precision (MAP)**: The MAP values also approached zero, reflecting the limitations in capturing user relevance due to dataset characteristics.

### Challenges & Learning Outcomes
- **Dataset Compatibility**: Finding datasets with similar column features for the training and test phases was challenging. This led to difficulties in building uniform models and performing accurate evaluations.
- **Preprocessing Complexity**: It took substantial effort to properly preprocess and align datasets, especially when including subjective analysis of review content.
- **Learning Outcomes**: The team gained deep insights into data exploration, the importance of preprocessing, and the nuances of collaborative filtering and matrix factorization models.

### How to Run the Project
To run this project, ensure you have the following dependencies installed:
- **Python** (>=3.7)
- **pandas** (for data manipulation)
- **Surprise** (for recommendation model implementation)
- **sklearn** (for data preprocessing and evaluation)
- **NumPy** (for numerical operations)
- **TfidfVectorizer** (for text feature extraction)

Follow these steps:
1. Clone the GitHub repository: [GitHub Repository](https://github.com/may0611/MachineLearning_Restaurant-RecommendingSystem)
2. Install the required libraries with `pip install -r requirements.txt`.
3. Run the preprocessing script to prepare data for modeling.
4. Execute the main recommendation script to train models and generate recommendations.
5. View the recommendations generated for users and analyze model performance metrics.

### Conclusion
The restaurant recommendation system successfully implemented collaborative filtering to provide personalized restaurant suggestions, albeit with certain limitations in precision due to dataset inconsistencies. Leveraging both numerical ratings and text features, the project showed the importance of data preprocessing and the challenges of merging datasets. The model produced meaningful and relevant results, achieving reasonable RMSE scores, which can be further refined by acquiring better datasets and enhancing preprocessing techniques.

### License
This project is licensed under the MIT License - see the LICENSE file for details.

### Acknowledgments
Special thanks to the Kaggle and Yelp teams for providing open datasets and to all project team members for their contributions and collaborative effort.
