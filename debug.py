# debug.py - NASCAR Driver Stats Analysis
import pandas as pd
import sys
from pathlib import Path

# Add project root and src to Python path
project_root = Path(__file__).parent
src_dir = project_root / 'src'
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(src_dir))

# Load the data
print("Loading NASCAR data...")
df = pd.read_parquet('data/raw/cup_series.parquet')  # or .csv

print("\n" + "="*50)
print("=== A.J. ALLMENDINGER ANALYSIS ===")
print("="*50)

# Filter for A.J. Allmendinger
aj_data = df[df['Driver'] == 'A.J. Allmendinger'].copy()

print(f"Total races for A.J. Allmendinger: {len(aj_data)}")
print(f"Wins (Finish == 1): {(aj_data['Finish'] == 1).sum()}")
print(f"Seasons: {sorted(aj_data['Season'].unique())}")

# Show all race results sorted by season/race
print("\nAll race results:")
aj_display = aj_data[['Season', 'Race', 'Track', 'Finish', 'Win']].copy()
aj_display = aj_display.sort_values(['Season', 'Race'])
print(aj_display.to_string())

# Manual calculations
finishes = aj_data['Finish'].dropna().values
total_races = len(finishes)
total_wins = (finishes == 1).sum()
top_5s = (finishes <= 5).sum()
top_10s = (finishes <= 10).sum()
avg_finish = finishes.mean()

print(f"\n=== MANUAL CALCULATIONS ===")
print(f"Total races: {total_races}")
print(f"Total wins: {total_wins}")
print(f"Total top-5s: {top_5s}")
print(f"Total top-10s: {top_10s}")
print(f"Average finish: {avg_finish:.3f}")
print(f"Win rate: {total_wins/total_races:.3f} ({total_wins/total_races:.1%})")
print(f"Top-5 rate: {top_5s/total_races:.3f} ({top_5s/total_races:.1%})")
print(f"Top-10 rate: {top_10s/total_races:.3f} ({top_10s/total_races:.1%})")

# Check processed data
print(f"\n=== PROCESSED SEASON DATA ===")
try:
    from data.data_loader import load_nascar_data
    
    loader = load_nascar_data()
    aj_seasons = loader.driver_seasons[loader.driver_seasons['Driver'] == 'A.J. Allmendinger']
    
    print("A.J. Allmendinger season summaries:")
    if not aj_seasons.empty:
        print(aj_seasons[['Season', 'races_run', 'wins', 'top_5s', 'top_10s', 'avg_finish', 'win_rate', 'top_5_rate', 'top_10_rate']].to_string())
        
        # Manual calculation from season data
        total_races_processed = aj_seasons['races_run'].sum()
        total_wins_processed = aj_seasons['wins'].sum()
        total_top5s_processed = aj_seasons['top_5s'].sum()
        total_top10s_processed = aj_seasons['top_10s'].sum()
        
        print(f"\n=== AGGREGATED FROM SEASONS ===")
        print(f"Total races: {total_races_processed}")
        print(f"Total wins: {total_wins_processed}")
        print(f"Total top-5s: {total_top5s_processed}")
        print(f"Total top-10s: {total_top10s_processed}")
        print(f"Aggregated win rate: {total_wins_processed/total_races_processed:.3f} ({total_wins_processed/total_races_processed:.1%})")
        print(f"Aggregated top-5 rate: {total_top5s_processed/total_races_processed:.3f} ({total_top5s_processed/total_races_processed:.1%})")
        print(f"Aggregated top-10 rate: {total_top10s_processed/total_races_processed:.3f} ({total_top10s_processed/total_races_processed:.1%})")
    else:
        print("No season data found for A.J. Allmendinger")
    
except Exception as e:
    print(f"Could not load processed data: {e}")

# Check API export data
print(f"\n=== API EXPORT DATA ===")
try:
    backend_data_dir = Path('backend/app/data')
    if backend_data_dir.exists():
        drivers_file = backend_data_dir / 'drivers.json'
        if drivers_file.exists():
            import json
            with open(drivers_file, 'r') as f:
                drivers_data = json.load(f)
            
            # Search for A.J. Allmendinger
            aj_found = False
            for driver in drivers_data.get('drivers', []):
                if 'allmendinger' in driver.get('name', '').lower():
                    print(f"Found in API data:")
                    print(f"Name: {driver.get('name')}")
                    stats = driver.get('career_stats', {})
                    print(f"API Total races: {stats.get('total_races')}")
                    print(f"API Total wins: {stats.get('total_wins')}")
                    print(f"API Career win rate: {stats.get('career_win_rate')} ({stats.get('career_win_rate', 0)*100:.1f}%)")
                    print(f"API Career top-5 rate: {stats.get('career_top5_rate')} ({stats.get('career_top5_rate', 0)*100:.1f}%)")
                    print(f"API Career top-10 rate: {stats.get('career_top10_rate')} ({stats.get('career_top10_rate', 0)*100:.1f}%)")
                    print(f"API Career avg finish: {stats.get('career_avg_finish')}")
                    aj_found = True
                    break
            
            if not aj_found:
                print("A.J. Allmendinger not found in exported API data")
        else:
            print("No drivers.json file found")
    else:
        print("Backend data directory doesn't exist")
            
except Exception as e:
    print(f"Could not check API export data: {e}")

print(f"\n=== COMPARISON SUMMARY ===")
print("This analysis compares:")
print("1. Manual calculations from raw race data")
print("2. Processed season-by-season data") 
print("3. Final API export data")
print("\nLook for discrepancies between these three sources!")
print("Expected: All three should match closely for the calculations to be correct.")