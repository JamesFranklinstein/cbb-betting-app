# 🏀 CBB Betting App

College Basketball betting analysis tool that combines **KenPom predictions** with **live betting odds** to identify value bets.

## Features

- **KenPom Integration** - Pull team ratings, predictions, and game projections
- **Live Odds** - Real-time odds from DraftKings, FanDuel, BetMGM, Caesars, and more
- **Value Detection** - Automatically find edges where KenPom disagrees with Vegas
- **All Bet Types** - Spreads, totals (over/under), and moneylines
- **Best Line Finder** - Shows which sportsbook has the best odds for each bet
- **ML Enhancement** - Machine learning model to improve predictions over time
- **CLI + Web Dashboard** - Use from terminal or browser

## Quick Start

### 1. Install Dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 2. Configure API Keys

```bash
cp .env.example .env
# Edit .env and add your API keys:
# KENPOM_API_KEY=your_key_here
# ODDS_API_KEY=your_key_here
```

### 3. Run the CLI

```bash
# See today's value bets
python cli.py today

# Analyze a specific matchup
python cli.py game "Duke" "North Carolina"

# Find spread discrepancies
python cli.py spreads --min-diff 2.0

# Find total discrepancies  
python cli.py totals --min-diff 3.0

# View KenPom rankings
python cli.py rankings --top 25
```

### 4. Run the Web Dashboard (Optional)

```bash
# Start the API server
cd backend
uvicorn api.main:app --reload --port 8000

# In another terminal, start the frontend
cd frontend
npm install
npm start
```

Open http://localhost:3000 in your browser.

## CLI Commands

| Command | Description |
|---------|-------------|
| `python cli.py today` | Show all today's games and value bets |
| `python cli.py today --min-edge 0.05` | Only show bets with 5%+ edge |
| `python cli.py today --all` | Show all games, not just value bets |
| `python cli.py game "Home" "Away"` | Analyze specific matchup |
| `python cli.py spreads` | Find KenPom vs Vegas spread differences |
| `python cli.py totals` | Find KenPom vs Vegas total differences |
| `python cli.py rankings` | Show current KenPom rankings |
| `python cli.py init` | Initialize the database |

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /api/games/today` | All games with analysis |
| `GET /api/value-bets?min_edge=0.03` | Today's value bets |
| `GET /api/matchup?home=Duke&away=UNC` | Specific game analysis |
| `GET /api/spreads?min_diff=2.0` | Spread discrepancies |
| `GET /api/totals?min_diff=3.0` | Total discrepancies |
| `GET /api/rankings?limit=50` | KenPom rankings |

## Understanding the Output

### Value Bet Example

```
📊 SPREAD: Duke -6.5 (-108)
   Model: 58.2% | Market: 52.4%
   Edge: 5.8% | EV: +4.2% | Kelly: 1.2%
   Best Book: DraftKings | Confidence: MEDIUM
```

- **Model**: Our predicted probability of covering
- **Market**: Vegas implied probability (from odds)
- **Edge**: Model probability - Market probability
- **EV**: Expected value per unit bet
- **Kelly**: Recommended bet size (% of bankroll)
- **Best Book**: Sportsbook with best odds

### Edge Thresholds

| Confidence | Spread/Total Edge | Moneyline Edge |
|------------|-------------------|----------------|
| High | 8%+ | 15%+ |
| Medium | 5-8% | 8-15% |
| Low | 3-5% | 5-8% |

## Project Structure

```
cbb-betting-app/
├── backend/
│   ├── api/
│   │   └── main.py          # FastAPI endpoints
│   ├── clients/
│   │   ├── kenpom.py        # KenPom API client
│   │   └── odds_api.py      # The Odds API client
│   ├── models/
│   │   ├── database.py      # SQLAlchemy models
│   │   └── connection.py    # DB connection
│   ├── services/
│   │   ├── game_service.py  # Game analysis logic
│   │   └── value_calculator.py  # Value bet detection
│   ├── ml/
│   │   └── predictor.py     # ML model
│   ├── cli.py               # Command-line interface
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   └── App.js           # React dashboard
│   └── package.json
└── .env.example
```

## How Value Detection Works

1. **Fetch KenPom Predictions**
   - Predicted scores for each team
   - Win probability
   - Expected tempo

2. **Fetch Live Odds**
   - Spreads, totals, moneylines from 7+ sportsbooks
   - Find best available odds for each side

3. **Compare & Calculate Edge**
   - Convert KenPom predictions to cover probabilities
   - Convert Vegas odds to implied probabilities
   - Edge = Our Probability - Vegas Probability

4. **Flag Value Bets**
   - Spread: 3%+ edge
   - Total: 3%+ edge
   - Moneyline: 5%+ edge

## ML Model (Optional)

The app includes an ML layer that can be trained on historical data:

```python
from ml import MLPredictor

predictor = MLPredictor()

# Train on historical data
metrics = predictor.train(training_data)

# Make predictions
result = predictor.predict_from_kenpom(home_rating, away_rating)
```

Features used:
- KenPom efficiency differentials (AdjEM, AdjOE, AdjDE)
- Four Factors (eFG%, TO%, OR%, FT Rate)
- Strength of schedule
- Home court advantage
- Recent form

## API Usage Notes

### KenPom API
- Bearer token authentication
- Rate limited (be respectful)
- Main endpoint: `/api.php?endpoint=fanmatch&d=YYYY-MM-DD`

### The Odds API
- API key as query parameter
- Usage quota tracked in response headers
- Cost: 1 credit per region per market
- Sport key: `basketball_ncaab`

## Deployment

### Railway (Recommended)

1. Push to GitHub
2. Connect Railway to your repo
3. Add environment variables
4. Deploy

### Docker

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY backend/requirements.txt .
RUN pip install -r requirements.txt
COPY backend/ .
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## Disclaimer

⚠️ **This tool is for informational and educational purposes only.**

- Past performance does not guarantee future results
- Sports betting involves risk of financial loss
- Gamble responsibly and within your means
- Check local laws regarding sports betting

## License

MIT
