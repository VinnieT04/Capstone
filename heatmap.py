# visualize_thesis_final.py  ← RUN THIS ONE
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the pivoted data (SSIDs are columns!)
df = pd.read_csv('Pivot_woAVG.csv')

# Remove the multi-index mess if present
if 'Unnamed: 0' in df.columns:
    df = df.drop(columns=['Unnamed: 0'])
if df.index.name == 'Scan_ID':
    df = df.reset_index()

# Get only the SSID columns (everything except Scan_ID, Location, Run)
ssid_columns = [col for col in df.columns if col not in ['Scan_ID', 'Location', 'Run']]

# Count how many unique locations each SSID appears in (non -100 values)
appearances = {}
for ssid in ssid_columns:
    # Count non -100 values per location
    counts = df[df[ssid] > -100].groupby('Location').size()
    appearances[ssid] = len(counts)

# Keep SSIDs that appear in at least 4 out of 5 zones
stable_ssids = [ssid for ssid, count in appearances.items() if count >= 4]
print(f"Stable SSIDs (≥4 zones): {len(stable_ssids)}")
print(stable_ssids)

# Compute mean RSSI per (Location, SSID), ignoring -100
mean_rssi = []
for ssid in stable_ssids:
    temp = df[df[ssid] > -100].groupby('Location')[ssid].mean().reset_index()
    temp['SSID'] = ssid
    mean_rssi.append(temp)
mean_df = pd.concat(mean_rssi, ignore_index=True)

# Shorten location names for beauty
loc_map = {
    'Hallway/LeftEnd': 'Left End',
    'LaptopPriority/CurrentPeriodicals': 'Laptop Area',
    'MainStairs/TopPicks': 'Main Stairs',
    'TopPicks/Audiovisual': 'Audiovisual',
    'Stairs/RightEnd': 'Right End'
}
mean_df['Location'] = mean_df['Location'].map(loc_map)

# Sort SSIDs by overall average strength
ssid_order = mean_df.groupby('SSID')[ssid].mean().sort_values(ascending=True).index[-21:]

# FINAL HEATMAP – YOUR STAR FIGURE
plt.figure(figsize=(10, 9))
plot_data = mean_df[mean_df['SSID'].isin(ssid_order)].pivot(index='SSID', columns='Location', values=ssid)

sns.heatmap(plot_data, annot=True, fmt=".1f", cmap="RdYlBu_r", linewidths=0.5,
            cbar_kws={'label': 'Average RSSI (dBm)'}, vmin=-90, vmax=-60)

plt.title('Wi-Fi Fingerprint Heatmap\n21 Most Reliable Access Points Across 5 Library Zones', 
          fontsize=16, fontweight='bold', pad=30)
plt.ylabel('Access Point (SSID)')
plt.xlabel('Library Zone')
plt.xticks(rotation=20, ha='right')
plt.tight_layout()

# SAVE FOR THESIS
plt.show()