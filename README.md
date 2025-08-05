# 🏁 NASCAR Driver Career Analysis

Ever wondered what makes some NASCAR drivers legends while others fade into the background? This project dives deep into 75+ years of NASCAR Cup Series data to uncover the hidden patterns in driver careers using machine learning.

## What's This All About?

I built this to answer questions like:
- **What "type" of driver is Kyle Larson?** (Spoiler: Elite Champion 🏆)
- **How do driver careers actually unfold?** (Think more complex than "good" vs "bad")
- **Can we predict a driver's future performance?** (Working on it with LSTM models!)
- **Who are the most similar drivers in NASCAR history?** (The clustering results might surprise you)

## The Cool Stuff

### 🤖 Machine Learning Magic
- **6 Driver Archetypes** discovered through K-means clustering:
  - Elite Champions (the GOATs like Gordon, Johnson)
  - Race Winners (solid performers like Kyle Busch)
  - Consistent Contenders (the reliable top-10 guys)
  - Veteran Journeymen (long careers, steady performance)
  - Mid-Pack Drivers (the NASCAR backbone)
  - Backfield Runners (hey, everyone starts somewhere)

### 📊 The Data
- **99,500+ race results** from 1949-2025
- **289 drivers** with 3+ seasons analyzed
- **Advanced feature engineering** (rolling averages, career phases, performance trends)
- **Real data** courtesy of the `nascaR.data` R package

### 🚀 Tech Stack
**Backend (Complete):**
- FastAPI serving ML models and data
- Python with scikit-learn, pandas, numpy
- Clustering analysis with silhouette optimization
- LSTM models for career trajectory prediction

**Frontend (In Progress):**
- React + TypeScript + Material-UI
- Plotly.js for interactive visualizations
- Modern dark theme with NASCAR orange accents
- Driver search, profile pages, archetype explorer

## Quick Start

### Backend API
```bash
# Get the data export running
python3 backend/scripts/export_data.py

# Fire up the API
cd backend
pip3 install -r requirements.txt
python3 -m uvicorn app.main:app --reload

# Check it out at http://localhost:8000/docs
```

### Try Some API Calls
```bash
# Search for drivers
curl "http://localhost:8000/api/drivers/search?q=kyle"

# Get Kyle Larson's profile
curl "http://localhost:8000/api/drivers/kyle-larson"

# See all the archetypes
curl "http://localhost:8000/api/archetypes"
```

## What You'll Find

### Driver Profiles
Each driver gets classified into one of the 6 archetypes with detailed career stats:
- Total wins, average finish, top-5 rate
- Career span and recent performance
- Similar drivers and archetype characteristics

### Archetype Analysis
- **Elite Champions** (21 drivers): The legends averaging 4+ wins per season
- **Race Winners** (29 drivers): Solid performers with 1+ wins per season
- **Consistent Contenders** (38 drivers): Your reliable top-10 finishers
- And 3 more distinct groups each with their own personality

### Interactive Comparisons
Compare any drivers side-by-side to see how their careers stack up across all the key metrics.

## The Nerdy Details

**Feature Engineering:**
- Rolling performance averages (3, 5, 10 race windows)
- Career phase detection (rookie, prime, veteran, decline)
- Equipment-adjusted metrics
- Consistency measurements (coefficient of variation)

**Clustering Approach:**
- K-means with 2-10 cluster evaluation
- Silhouette score optimization
- Feature scaling and selection
- Representative driver identification

**LSTM Predictions:**
- Sequential modeling of career trajectories
- Multi-step ahead forecasting
- Performance confidence intervals
- Career milestone predictions

## What's Next?

- 🎨 **React Frontend**: Beautiful, interactive driver explorer
- 🧠 **Enhanced LSTM**: Better career predictions with uncertainty quantification
- 📈 **More Visualizations**: Career trajectory plots, archetype comparisons
- 🏆 **Championship Probability**: Who's most likely to win it all?
- 📱 **Mobile App**: Because race day needs data

## Fun Facts I Discovered

- Richard Petty isn't just the "King" - he's statistically in a league of his own
- There are actually 3 distinct types of "champions" in NASCAR history
- Some drivers peak early, others are late bloomers - and the data shows exactly when
- Modern NASCAR has more parity than you might think (looking at you, archetype distribution)

## Contributing

This started as a portfolio project but has become genuinely useful for NASCAR analysis. If you're interested in motorsports analytics, machine learning, or just think this is cool, feel free to:

- Open issues for bugs or feature requests
- Submit PRs for improvements
- Share interesting findings from the data
- Suggest new analysis approaches

## Data Source

Huge thanks to the `nascaR.data` R package maintainers for providing clean, comprehensive NASCAR data. This project wouldn't exist without their work making NASCAR statistics accessible.

---

*Built with ☕ and a love for both data science and left turns at 200mph.*