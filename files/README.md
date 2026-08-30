# Adaptive Learning System - Frontend

A basic HTML/CSS/JS frontend for the Adaptive Learning System backend.

## Files

- **index.html** - Main HTML file with login, dashboard, and quiz pages
- **style.css** - Basic styling
- **script.js** - JavaScript logic for handling quiz, stats, and API calls

## How to Use

### 1. Start the Backend Services

Make sure both backend services are running:

```bash
# Node.js Backend (port 4000)
node server.js

# Python ML Service (port 8000)
python main.py
```

### 2. Open the Frontend

Simply open `index.html` in your web browser:
- Right-click on `index.html` → Open with Browser
- Or use a local server: `python -m http.server 8000`

### 3. Using the App

1. **Login Page**: Enter your name to start
2. **Dashboard**: View your stats and start a quiz
3. **Quiz**: Answer questions, the difficulty adjusts based on your performance
4. **Results**: See if you got it right and what the next difficulty level is

## Features

- ✓ Basic login system
- ✓ Dashboard with statistics (total questions, accuracy, current level)
- ✓ Quiz interface with multiple choice questions
- ✓ Integrates with your ML service to predict next difficulty
- ✓ Tracks user performance metrics
- ✓ Different question banks for easy/medium/hard levels

## Configuration

Edit the `API_URL` in `script.js` if your backend is running on a different address:

```javascript
const API_URL = 'http://localhost:4000';
```

## How It Works

1. User enters their name on login page
2. Dashboard shows current stats (total questions, accuracy, difficulty level)
3. User clicks "Start Quiz" to begin
4. Questions are presented based on current difficulty level
5. When user submits an answer:
   - Frontend calculates: correctness, time taken
   - Calls `/question/submit` endpoint with performance data
   - Receives next difficulty prediction from ML service
   - Updates user stats and moves to next question
6. Difficulty adapts automatically based on ML service response

## API Endpoints Used

- `POST /question/submit` - Submit an answer and get difficulty prediction
  - Request: `{ isCorrect, timeTaken, attempts, pastAccuracy }`
  - Response: `{ nextDifficulty }`
