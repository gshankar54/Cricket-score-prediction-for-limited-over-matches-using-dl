from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import Huber

app = Flask(__name__)

# Global variables for encoders and scaler
venue_encoder = LabelEncoder()
batting_team_encoder = LabelEncoder()
bowling_team_encoder = LabelEncoder()
striker_encoder = LabelEncoder()
bowler_encoder = LabelEncoder()
match_type_encoder = LabelEncoder()
scaler = MinMaxScaler()

# Combine datasets
def load_combined_data():
    ipl_data = pd.read_csv("D:\\api\\ipl_data.csv")
    t20_data = pd.read_csv("D:\\api\\t20.csv")
    odi_data = pd.read_csv("D:\\api\\odi.csv")

    # Add match_type column
    ipl_data['match_type'] = 'IPL'
    t20_data['match_type'] = 'T20'
    odi_data['match_type'] = 'ODI'

    # Combine all datasets
    combined_data = pd.concat([ipl_data, t20_data, odi_data], ignore_index=True)
    return combined_data

# Training data and model initialization
def prepare_model():
    global venue_encoder, batting_team_encoder, bowling_team_encoder, striker_encoder, bowler_encoder, match_type_encoder, scaler

    # Load combined dataset
    df = load_combined_data()

    # Preprocess the data
    df = df.drop(['date', 'runs_last_5', 'wickets_last_5', 'mid', 'striker', 'non-striker'], axis=1)
    X = df.drop(['total'], axis=1)
    y = df['total']

    # Encode categorical features
    X['venue'] = venue_encoder.fit_transform(X['venue'])
    X['bat_team'] = batting_team_encoder.fit_transform(X['bat_team'])
    X['bowl_team'] = bowling_team_encoder.fit_transform(X['bowl_team'])
    X['batsman'] = striker_encoder.fit_transform(X['batsman'])
    X['bowler'] = bowler_encoder.fit_transform(X['bowler'])
    X['match_type'] = match_type_encoder.fit_transform(X['match_type'])

    # Include only relevant features
    X = X[['venue', 'bat_team', 'bowl_team', 'batsman', 'bowler', 'match_type', 'wickets', 'overs']]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Fit scaler
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Define and train the model
    model = Sequential([
        Dense(512, activation='relu', input_shape=(X_train_scaled.shape[1],)),
        Dense(216, activation='relu'),
        Dense(1, activation='linear')
    ])
    huber_loss = Huber(delta=1.0)
    model.compile(optimizer='adam', loss=huber_loss)
    model.fit(X_train_scaled, y_train, epochs=5, batch_size=128, validation_data=(X_test_scaled, y_test))

    return model, df


# Train the model
model, df = prepare_model()

@app.route('/')
def index():
    return render_template('index.html', 
                           venues=df['venue'].unique().tolist(),
                           teams=df['bat_team'].unique().tolist(),
                           players=df['batsman'].unique().tolist(),
                           bowlers=df['bowler'].unique().tolist(),
                           match_types=df['match_type'].unique().tolist())

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    try:
        # Encode the inputs
        venue = venue_encoder.transform([data['venue']])[0]
        batting_team = batting_team_encoder.transform([data['batting_team']])[0]
        bowling_team = bowling_team_encoder.transform([data['bowling_team']])[0]
        striker = striker_encoder.transform([data['striker']])[0]
        bowler = bowler_encoder.transform([data['bowler']])[0]
        match_type = match_type_encoder.transform([data['match_type']])[0]
        
        # Get numerical inputs
        wickets = float(data['wickets'])
        overs = float(data['overs'])

        # Create input array and scale it
        input_data = np.array([[venue, batting_team, bowling_team, striker, bowler, match_type, wickets, overs]])
        input_data_scaled = scaler.transform(input_data)

        # Predict
        predicted_score = model.predict(input_data_scaled)
        predicted_score = int(predicted_score[0, 0])
        return jsonify({'predicted_score': predicted_score})

    except Exception as e:
        return jsonify({'error': str(e)})



if __name__ == '__main__':
    app.run(debug=True)
