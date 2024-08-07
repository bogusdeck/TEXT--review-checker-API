import pickle
import os
import string
import pandas as pd
import numpy as np
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

nltk.download("stopwords")
load_dotenv()

app = Flask(__name__)
CORS(app)
model = pickle.load(open("/api/model.pkl", "rb"))


cohere_api_key = os.getenv("COHERE_API_KEY")
if not cohere_api_key:
    raise ValueError(
        "Cohere API key not found. Please set the COHERE_API_KEY environment variable."
    )

llm = ChatCohere(cohere_api_key=cohere_api_key)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "I want you to check the authenticity of the review written below and check if the given product review detail is matching with other details and also check the validity of the text review that it is worth enough to give reward points and I want the answer in one word 'Original' or 'Fake' ",
        ),
        ("user", "{input}"),
    ]
)

output_parser = StrOutputParser()

chain = prompt | llm | output_parser

df = pd.read_csv("deceptive-opinion.csv")

null_values = df.isnull()

columns_with_none = null_values.any()

print("Columns with None values:")
print(columns_with_none)



df1 = df[["deceptive", "text"]].copy() 
df1.loc[df1["deceptive"] == "deceptive", "deceptive"] = 0
df1.loc[df1["deceptive"] == "truthful", "deceptive"] = 1

df1.dropna(subset=["text"], inplace=True)

X = df1["text"]
Y = np.asarray(df1["deceptive"], dtype=int)

X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.3, random_state=109
)


def text_process(review):
    if review is not None:
        nopunc = [char for char in review if char not in string.punctuation]
        nopunc = "".join(nopunc)
        return [
            word
            for word in nopunc.split()
            if word.lower() not in stopwords.words("english")
        ]


cv = CountVectorizer()
X_train_cv = cv.fit_transform(X_train)


X_test_cv = cv.transform(X_test)

nb = MultinomialNB()
nb.fit(X_train_cv, y_train)

clf = svm.SVC(kernel="linear")
clf.fit(X_train_cv, y_train)

if not os.path.exists("model.pkl"):
    with open("model.pkl", "wb") as f:
        pickle.dump((model, cv), f)


@app.route("/api/review", methods=["POST"])
def predict():

    data = request.json
    print(data)
    # data = [product_review]

    name = data["productName"]
    category = data["category"]
    brand = data["brand"]
    purchase_date = data["purchaseDate"]
    purchase_price = data["purchasePrice"]
    product_review = data["productReview"]
    shoppingLink = data["shoppingLink"]

    data_list = [product_review]

    input_text = f"I bought the {name} {category} of {brand} from {shoppingLink} on {purchase_date} at {purchase_price}. This is my review about it: {product_review}"

    response = chain.invoke({"input": input_text})

    print(response)

    with open("model.pkl", "rb") as f:
        model = pickle.load(f)

    print("here is the model:", model)
    vect = cv.transform(data_list).toarray()

    prediction = model.predict(vect)

    print("here is the prediction", prediction)

    my_response = "Original" if prediction == [1] else "Fake"


    if response == "Original" and my_response == "Original":
        message = "Original"
    else:
        message = "Fake"

    return jsonify({"message": message})


if __name__ == "__main__":
    app.run(debug=True)

