import numpy as np
import pandas as pd
from sklearn.linear_model import PassiveAggressiveRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer

class ReachPredictor:
    def __init__(self):
        self.model = PassiveAggressiveRegressor(max_iter=1000, random_state=42, tol=1e-3)
        self.scaler = StandardScaler()
        self.tfidf = TfidfVectorizer(max_features=50)
        self.numeric_features = None  # Placeholder for numeric features count
    
    def train(self, data_path: str):
        # Load the dataset
        data = pd.read_csv(data_path, encoding='latin1')
        
        # Feature engineering
        numeric_features = data[['Likes', 'Saves', 'Comments', 'Shares', 'Profile Visits', 'Follows']].values
        hashtag_features = self.tfidf.fit_transform(data['Hashtags']).toarray()
        self.numeric_features = numeric_features.shape[1]
        
        X = np.hstack([numeric_features, hashtag_features])
        y = data['Impressions'].values  # Target variable
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Normalize numeric features
        X_train[:, :self.numeric_features] = self.scaler.fit_transform(X_train[:, :self.numeric_features])
        X_test[:, :self.numeric_features] = self.scaler.transform(X_test[:, :self.numeric_features])
        
        # Train the model
        self.model.fit(X_train, y_train)
        
        # Evaluate the model
        y_pred = self.model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2_score = self.model.score(X_test, y_test)
        
        return {
            "MSE": mse,
            "RMSE": rmse,
            "R2": r2_score
        }

    def predict(self, input_data: dict):
        # Process numeric inputs
        numeric_inputs = np.array([[input_data['Likes'], input_data['Saves'], input_data['Comments'], 
                                    input_data['Shares'], input_data['Profile Visits'], input_data['Follows']]])
        numeric_inputs = self.scaler.transform(numeric_inputs)
        
        # Process hashtags
        hashtag_inputs = self.tfidf.transform([input_data['Hashtags']]).toarray()
        
        # Combine numeric and hashtag inputs
        input_features = np.hstack([numeric_inputs, hashtag_inputs])
        
        # Predict impressions
        return self.model.predict(input_features)[0]
