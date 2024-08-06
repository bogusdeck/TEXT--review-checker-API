import pickle
import os
import string
import pandas as pd
import numpy as np
# import pywedge as pw
import sklearn as sk
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.model_selection import train_test_split
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
import nltk
from nltk.corpus import stopwords
from langchain_cohere import ChatCohere
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

# Download stopwords data
nltk.download("stopwords")
load_dotenv()

# Define Flask app
app = Flask(__name__)
CORS(app)
# # Load the pre-trained model
model = pickle.load(open("model.pkl", "rb"))


cohere_api_key = os.getenv("COHERE_API_KEY")
if not cohere_api_key:
    raise ValueError("Cohere API key not found. Please set the COHERE_API_KEY environment variable.")

llm = ChatCohere(cohere_api_key = cohere_api_key)

prompt = ChatPromptTemplate.from_messages([
    ("system", "I want you to check the authenticity of the review written below and check if the given product review detail is matching with other details and also check the validity of the text review that it is worth enough to give reward points and I want the answer in one word 'Original' or 'Fake' "),
    ("user", "{input}")
])

output_parser = StrOutputParser()

chain = prompt | llm | output_parser

# Load the dataset
df = pd.read_csv("deceptive-opinion.csv")

# Check for None values in the DataFrame
null_values = df.isnull()

# Check if there are any None values in each column
columns_with_none = null_values.any()

# Print the columns with None values
print("Columns with None values:")
print(columns_with_none)

# # Optionally, you can print the rows with None values as well
# rows_with_none = df[df.isnull().any(axis=1)]
# print("Rows with None values:")
# print(rows_with_none)


# Preprocess the dataset
df1 = df[["deceptive", "text"]].copy()  # Create a copy of the DataFrame
df1.loc[df1["deceptive"] == "deceptive", "deceptive"] = 0
df1.loc[df1["deceptive"] == "truthful", "deceptive"] = 1

# Remove rows with missing values
df1.dropna(subset=["text"], inplace=True)

# Split the dataset into features and target
X = df1["text"]
Y = np.asarray(df1["deceptive"], dtype=int)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.3, random_state=109
)

# Initialize CountVectorizer with custom text processing function
def text_process(review):
    if review is not None:
        nopunc = [char for char in review if char not in string.punctuation]
        nopunc = "".join(nopunc)
        return [
            word
            for word in nopunc.split()
            if word.lower() not in stopwords.words("english")
        ]

# # Initialize CountVectorizer with custom text processing function
# cv = CountVectorizer(analyzer=text_process)
# x_train_cv = cv.fit_transform(X_train)
# x_test_cv = cv.transform(X_test)

# During training
cv = CountVectorizer()
X_train_cv = cv.fit_transform(X_train)
# Train your model using X_train_cv


# Use the same CountVectorizer instance used during training
X_test_cv = cv.transform(X_test)
# Make predictions using the model

# Train Naive Bayes model
nb = MultinomialNB()
nb.fit(X_train_cv, y_train)

# Train SVM model
clf = svm.SVC(kernel="linear")
clf.fit(X_train_cv, y_train)

# Save the model and CountVectorizer if the file doesn't exist
if not os.path.exists("model.pkl"):
    with open("model.pkl", "wb") as f:
        pickle.dump((model, cv), f)

@app.route("/api/review", methods=["POST"])
def predict():
    # product_name = request.form["productName"]
    # category = request.form["category"]
    # brand = request.form["brand"]
    # purchase_date = request.form["purchaseDate"]
    # purchase_price = request.form["purchasePrice"]
    # product_review = request.form["productReview"]

    data = request.json
    print(data)
    # data = [product_review]
    
    name = data['productName']
    category = data['category']
    brand = data['brand']
    purchase_date = data['purchaseDate']
    purchase_price = data['purchasePrice']
    product_review = data['productReview']
    shoppingLink = data['shoppingLink']

    data_list = [product_review]

    input_text = f"I bought the {name} {category} of {brand} from {shoppingLink} on {purchase_date} at {purchase_price}. This is my review about it: {product_review}"    

    response = chain.invoke({"input":input_text})

    print(response)
    
    # During prediction
    # with open("model.pkl", "rb") as f:
    # model, cv = pickle.load(f)

    # Use the CountVectorizer instance to transform the input data
    # vect = cv.transform(data).toarray()

    # Make predictions using the model
    # prediction = model.predict(vect)

    # During prediction
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)

    print("here is the model:",model)
    # Use the CountVectorizer instance to transform the input data
    vect = cv.transform(data_list).toarray()

    # Make predictions using the model
    prediction = model.predict(vect)

    print("here is the prediction", prediction)

    my_response = "Original" if prediction == [1] else "Fake"

    # if response == "Original" and (my_response=="Original" or my_response=="Fake"): 
    #     message = "Original"
    # else:
    #     message = "Fake"
    
    if response == "Original" and my_response=="Original": 
        message = "Original"
    else:
        message = "Fake"
    
    return jsonify({"message": message})

if __name__ == "__main__":
    app.run(debug=True)