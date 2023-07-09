import json
import joblib

def news_classification(text):
    """
    Given sepal length, sepal width, petal length, and petal width,
    predict the class of iris
    """
    
    # Load the model from the file
    with open("model.pkl", "rb") as f:
        baseline_model = joblib.load(f)

    # Vectorize the new text using the same vectorizer
    new_text_vectorized = tfidf.transform([text])

    # Make predictions using the trained model
    predicted_label = baseline_model.predict(new_text_vectorized)
    
    return {"predicted_label": predicted_label}

def predict(request):
    """
    `request` is an HTTP request object that will automatically be passed
    in by Google Cloud Functions
    
    You can find all of its properties and methods here:
    https://flask.palletsprojects.com/en/1.0.x/api/#flask.Request
    """
    
    # Get the request data from the user in JSON format
    request_json = request.get_json()
    
    # Send it to our prediction function using ** to unpack the arguments
    result = news_classification(**request_json)
    
    # Return the result as a string with JSON format
    return json.dumps(result)