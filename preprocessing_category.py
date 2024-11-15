import pandas as pd


# Load the uploaded files
review_noun = pd.read_csv("C:\\studyMachineLearning\\dataSet\\pos_tag.csv")
review_adjective = pd.read_csv("C:\\studyMachineLearning\\dataSet\\review_adjectives2.csv")

# Define keywords for each category based on common themes in reviews
service_keywords = ['friendly', 'attentive', 'staff', 'service', 'wait']
food_keywords = ['delicious', 'phenomenal', 'tasty', 'bite', 'crawfish', 'flavor', 'taste']
ambiance_keywords = ['nicely', 'decorate', 'place', 'ambiance', 'setting']
value_keywords = ['expensive', 'affordable', 'worth', 'cost', 'price']
experience_keywords = ['good', 'great', 'unhappy', 'favorite', 'love', 'satisfy', 'dissatisfy']

# Function to categorize reviews based on keywords
def categorize_review(nouns, adjectives):
    categories = {
        'Service Quality': 0,
        'Food Quality': 0,
        'Ambiance': 0,
        'Value for Money': 0,
        'Overall Experience': 0
    }
    
    # Check for matches in nouns
    for word in nouns:
        if word in service_keywords:
            categories['Service Quality'] += 1
        if word in food_keywords:
            categories['Food Quality'] += 1
        if word in ambiance_keywords:
            categories['Ambiance'] += 1
        if word in value_keywords:
            categories['Value for Money'] += 1
        if word in experience_keywords:
            categories['Overall Experience'] += 1

    # Check for matches in adjectives
    for word in adjectives:
        if word in service_keywords:
            categories['Service Quality'] += 1
        if word in food_keywords:
            categories['Food Quality'] += 1
        if word in ambiance_keywords:
            categories['Ambiance'] += 1
        if word in value_keywords:
            categories['Value for Money'] += 1
        if word in experience_keywords:
            categories['Overall Experience'] += 1
            
    # Assign category with the highest count
    max_category = max(categories, key=categories.get)
    return max_category

# Apply categorization to each review
# Merge dataframes on common columns for easier categorization
merged_df = pd.merge(review_noun, review_adjective, on=['review_id', 'user_id', 'business_id', 'stars'], suffixes=('_nouns', '_adjectives'))
merged_df['Category'] = merged_df.apply(lambda row: categorize_review(eval(row['text_nouns']), eval(row['text_adjectives'])), axis=1)

# Display the categorized reviews to the user
print(merged_df)

# To save as a CSV file
merged_df.to_csv("review_category.csv", index=False)
print("Data saved as 'review_category.csv'. Open this file in a spreadsheet editor to view.")