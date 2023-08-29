#IMPORT NECESSARY LIBRARIES
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string

# Load the dataset
bot_dataset = pd.read_csv("C:\Users\user\Desktop\Simple chat_bot\topical_chat.csv")
bot_dataset.head()

'''print(bot_dataset.info())
print(bot_dataset["sentiment"].value_counts())'''

# Download stopwords and punkt tokenizer
nltk.download('punkt')
nltk.download('stopwords')

# Preprocessing function
def preprocess_text(text):
    # Tokenize
    tokens = word_tokenize(text)

    # Remove stopwords and punctuation
    tokens = [word.lower() for word in tokens if word.isalnum() and word.lower() not in stopwords.words("english")]

    return " ".join(tokens)

# Apply preprocessing to the "message" column
bot_dataset["processed_message"] = bot_dataset["message"].apply(preprocess_text)

####
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

X = bot_dataset["processed_message"]
y = bot_dataset["sentiment"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Convert text data to numerical data
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

#Encode the sentiment label using LabelEncode()
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

#Tokenize and pad the text data for input to the neural network
tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)

X_train_sequences = tokenizer.texts_to_sequences(X_train)
X_test_sequences = tokenizer.texts_to_sequences(X_test)

X_train_padded = pad_sequences(X_train_sequences, maxlen=100, padding="post", truncating="post")
X_test_padded = pad_sequences(X_test_sequences, maxlen=100, padding="post", truncating="post")

#BUILDING THE NEURAL NETWORK
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

model = Sequential([
    Embedding(input_dim=5000, output_dim=128, input_length=100),
    LSTM(128, return_sequences=True),
    LSTM(64),
    Dense(64, activation='relu'),
    #Dropout(0.5),
    Dense(8, activation='linear')
])

# Compile the model
model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.Adam(0.001),
    metrics=['accuracy']
)
#Print the model summary
model.summary()

#TRAIN THE MODEL
model.fit(X_train_padded, y_train_encoded, epochs=5, batch_size=45, validation_split=0.1)

#Evaluate the model
loss, accuracy = model.evaluate(X_test_padded, y_test_encoded)
print("Test accuracy:", accuracy)


def predict_sentiment(text):
    processed_text = preprocess_text(text)
    sequence = tokenizer.texts_to_sequences([processed_text])
    padded_sequence = pad_sequences(sequence, maxlen=100, padding="post", truncating="post")
    sentiment_probabilities = model.predict(padded_sequence)
    predicted_sentiment_id = np.argmax(sentiment_probabilities)
    predicted_sentiment = label_encoder.inverse_transform([predicted_sentiment_id])[0]
    return predicted_sentiment

user_input = input("Enter a message: ")
predicted_sentiment = predict_sentiment(user_input)
print("Predicted sentiment:", predicted_sentiment)


def generate_rule_based_response(predicted_sentiment):
    if predicted_sentiment == "Happy":
        response = "I'm glad to hear that you're feeling happy!"
    elif predicted_sentiment == "Sad":
        response = "I'm sorry to hear that you're feeling sad. Is there anything I can do to help?"
    else:
        response = "I'm here to chat with you. How can I assist you today?"

    return response

def generate_rule_based_response_chatbot(user_input):
    # Predict sentiment using your neural network model (code you've shared earlier)
    predicted_sentiment = predict_sentiment_nn(user_input)

    # Generate response based on predicted sentiment using rule-based approach
    response = generate_rule_based_response(predicted_sentiment)

    return response

def generate_pattern_response(user_input):
    patterns = {
        "hello": "Hello! How can I assist you today?",
        "how are you": "I'm just a chatbot, but I'm here to help! How can I assist you?",
        "help": "Sure, I'd be happy to help. What do you need assistance with?",
        "bye": "Goodbye! If you have more questions in the future, feel free to ask.",
        # Add more patterns and responses here
    }

    # Look for pattern matches and return the corresponding response
    for pattern, response in patterns.items():
        if pattern in user_input.lower():
            return response

    # If no pattern matches, use the rule-based response based on sentiment
    return generate_rule_based_response_chatbot(user_input)

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        print("Bot: Goodbye!")
        break
    bot_response = generate_pattern_response(user_input)
    print("Bot:", bot_response)
