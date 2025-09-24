import pandas as pd
from pathlib import Path

# --- 1) Variablenliste laden  
var_list_path = "./data/ess11_full_variables.csv"
var_df = pd.read_csv(var_list_path)
selected_vars = var_df["var_name"].dropna().unique().tolist()

# --- 2) ESS-Daten aus CSV laden 
csv_path = "./data/ESS11.csv"   
df = pd.read_csv(csv_path)

# --- 2.1) Spezifische Länder auswählen 
selected_countries = [ "PL", "SE", "GB", "IT", "ES", "HU", "NL"]
print(f"Lade Daten für Länder: {selected_countries}")

# Filtere nach den gewünschten Ländern
country_filtered = df[df["cntry"].isin(selected_countries)].copy()
print(f"Gefunden: {len(country_filtered)} Personen in den ausgewählten Ländern")

# Verteilung nach Ländern
country_counts = country_filtered["cntry"].value_counts()
print("Verteilung nach Ländern:")
for country, count in country_counts.items():
    print(f"  {country}: {count} Personen")

# --- 2.2) Zufällige Stichprobe: 300 Personen pro Land 
import numpy as np

# Seed für reproduzierbare Ergebnisse
np.random.seed(42)

sampled_data = []
persons_per_country = 300

print(f"\nZiehe zufällig {persons_per_country} Personen pro Land...")

for country in selected_countries:
    country_data = country_filtered[country_filtered["cntry"] == country]
    
    if len(country_data) >= persons_per_country:
        # Zufällige Stichprobe ziehen
        sampled = country_data.sample(n=persons_per_country, random_state=42)
        sampled_data.append(sampled)
        print(f"  {country}: {len(sampled)} Personen ausgewählt (von {len(country_data)} verfügbar)")
    else:
        # Falls weniger als 300 verfügbar, alle nehmen
        sampled_data.append(country_data)
        print(f"  {country}: {len(country_data)} Personen ausgewählt (nur {len(country_data)} verfügbar)")

# Alle Stichproben zusammenführen
df = pd.concat(sampled_data, ignore_index=True)
print(f"\nGesamt: {len(df)} Personen aus {len(selected_countries)} Ländern")

# --- 3) Immer behalten: ID, Land, Gewichte (nur falls vorhanden) 
always_keep = ["idno", "cntry", "dweight", "pspwght", "pweight", "anweight"]
always_keep = [c for c in always_keep if c in df.columns]

# --- 4) Nur existierende Zielvariablen behalten + Always-keep zusammenführen 
existing_vars = [v for v in selected_vars if v in df.columns]
keep_cols = list(dict.fromkeys(always_keep + existing_vars))  # Reihenfolge bewahren, Duplikate entfernen
filtered_df = df[keep_cols].copy()

# --- 5) Eindeutige ID bauen (cntry + idno) 
if "cntry" in filtered_df.columns and "idno" in filtered_df.columns:
    filtered_df["unique_id"] = (
        filtered_df["cntry"].astype(str).str.strip()
        + "_"
        + filtered_df["idno"].astype(str).str.replace(r"\.0$", "", regex=True).str.strip()
    )
    # Optional: unique_id nach vorne ziehen
    front = ["unique_id", "cntry", "idno"]
    front = [c for c in front if c in filtered_df.columns]
    other = [c for c in filtered_df.columns if c not in front]
    filtered_df = filtered_df[front + other]

# --- 6) Speichern 
output_csv = "./data/ess11_full_filtered.csv"
output_pkl = "./data/ess11_full_filtered.pkl"

filtered_df.to_csv(output_csv, index=False)
filtered_df.to_pickle(output_pkl)

print(f"\n✔️ Gefilterter Datensatz gespeichert:")
print(f"   CSV: {output_csv}")
print(f"   PKL: {output_pkl}")
print(f"   Spalten: {filtered_df.shape[1]}")
print(f"   Zeilen: {filtered_df.shape[0]}")

# Finale Verteilung nach Ländern
final_counts = filtered_df["cntry"].value_counts().sort_index()
print(f"\nFinale Verteilung nach Ländern:")
for country in selected_countries:
    count = final_counts.get(country, 0)
    print(f"  {country}: {count} Personen")

print(f"\nErwartet: {len(selected_countries)} × 300 = {len(selected_countries) * 300} Personen")
print(f"Tatsächlich: {len(filtered_df)} Personen")
print("Columns:", filtered_df.columns.tolist()[:20], "...")





