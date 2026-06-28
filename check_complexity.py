import pandas as pd
import textstat

FILE_IN = 'risultati_hotpot_500.csv'
FILE_OUT = 'report_analisi_4_indicatori.csv'

def compute_indicators(text):
    t = str(text).replace('"', '').replace("'", "").strip()
    if len(t) < 3:
        return 0.0, 0.0, 0.0, 0.0
    # 1. SMOG
    smog = textstat.smog_index(t)
    # 2. DALE-CHALL
    dale = textstat.dale_chall_readability_score(t)
    # 3. GUNNING FOG
    fog = textstat.gunning_fog(t)
    # 4. FLESCH-KINCAID GRADE
    flesch = textstat.flesch_kincaid_grade(t)
    return smog, dale, fog, flesch

# Load CSV (semicolon separator, consistent with other scripts in the project)
try:
    df = pd.read_csv(FILE_IN, sep=';', on_bad_lines='skip')
except FileNotFoundError:
    print(f"ERROR: file '{FILE_IN}' not found.")
    exit()

# Check that required columns exist
required_columns = ['Clear_Prompt', 'Complex_Prompt']
missing = [c for c in required_columns if c not in df.columns]
if missing:
    print(f"ERROR: missing columns in CSV: {missing}")
    print(f"Available columns: {list(df.columns)}")
    exit()

metrics = ['Smog', 'DaleChall', 'Fog', 'Flesch']

df[[f'{m}_Clear' for m in metrics]] = df['Clear_Prompt'].apply(
    lambda x: pd.Series(compute_indicators(x))
)
df[[f'{m}_Complex' for m in metrics]] = df['Complex_Prompt'].apply(
    lambda x: pd.Series(compute_indicators(x))
)

for m in metrics:
    df[f'Delta_{m}'] = df[f'{m}_Complex'] - df[f'{m}_Clear']
    anomalies = df[df[f'Delta_{m}'] < 0]
    if len(anomalies) > 0:
        print(f"\n" + "!"*60)
        print(f"WARNING [Anomaly {m}]: {len(anomalies)} cases where the Clear prompt scores as more complex.")
        print("!"*60)
        for index, row in anomalies.iterrows():
            print(f"Row: {index} | Delta {m}: {row[f'Delta_{m}']:.2f}")
            print(f"CLEAR  : {row['Clear_Prompt']}")
            print(f"COMPLEX: {row['Complex_Prompt']}")
            print("-" * 60)

df.round(2).to_csv(FILE_OUT, index=False, sep=';', decimal=',')

print(f"\n{'METRIC':<15} | {'CLEAR':<8} | {'COMPLEX':<8} | MEANING")
print("-" * 80)
print(f"{'SMOG Index':<15} | {df['Smog_Clear'].mean():<8.2f} | {df['Smog_Complex'].mean():<8.2f} | Ambiguity")
print(f"{'Dale-Chall':<15} | {df['DaleChall_Clear'].mean():<8.2f} | {df['DaleChall_Complex'].mean():<8.2f} | Vocabulary difficulty (rare words)")
print(f"{'Gunning Fog':<15} | {df['Fog_Clear'].mean():<8.2f} | {df['Fog_Complex'].mean():<8.2f} | Reading effort")
print(f"{'Flesch Grade':<15} | {df['Flesch_Clear'].mean():<8.2f} | {df['Flesch_Complex'].mean():<8.2f} | Years of education required")
print("="*80)
