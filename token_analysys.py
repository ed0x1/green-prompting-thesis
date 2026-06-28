import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

FILE = 'risultati_hotpot_500.csv'   # 🆕 nome del tuo file

try:
    df = pd.read_csv(FILE, sep=';', on_bad_lines='skip')
except FileNotFoundError:
    print(f"ERROR: File '{FILE}' not found.")
    exit()

print(f"Loaded {len(df)} rows from '{FILE}'.")

# --- TOKEN ANALYSIS ---
df = df[['Tokens_Clear_Std', 'Tokens_Clear_Short', 'Tokens_Complex_Std', 'Tokens_Complex_Short']]
df.columns = ['Clear', 'Clear_Short', 'Complex', 'Complex_Short']
means = df.mean()

style_savings      = (means.Complex - means.Clear)             / means.Complex       * 100
constraint_savings = (means.Complex - means.Complex_Short)     / means.Complex       * 100
extra_savings      = (means.Complex_Short - means.Clear_Short) / means.Complex_Short * 100
combined_effect    = (means.Complex - means.Clear_Short)       / means.Complex       * 100

print("\n=== TOKEN ANALYSIS ===")
print(f"1. Style Savings:      -{style_savings:.1f}%   (Complex -> Clear)")
print(f"2. Constraint Savings: -{constraint_savings:.1f}%   (Complex -> Complex_Short)")
print(f"3. Extra Savings:      -{extra_savings:.1f}%   (Complex_Short -> Clear_Short)")
print(f"4. Combined Effect:    -{combined_effect:.1f}%   (Complex -> Clear_Short)")

# --- PLOTS ---
sns.set_theme(style="whitegrid")
ordered_cols = ['Complex', 'Complex_Short', 'Clear', 'Clear_Short']

# Boxplot
plt.figure(figsize=(8, 5))
sns.boxplot(
    data=df[ordered_cols],
    color="white",
    linecolor="black",
    linewidth=1.2,
    flierprops=dict(marker='o', color='black', markeredgecolor='black', alpha=0.5)
)
plt.title('Generated Tokens Distribution', fontsize=14, fontweight='bold', pad=15)
plt.ylabel('Number of Tokens', fontsize=12)
plt.tight_layout()
plt.savefig('1_boxplot_distribution_bw.png')
plt.close()

# Bar chart
plt.figure(figsize=(6, 4))
summary = pd.Series(
    [means.Complex, means.Complex_Short, means.Clear, means.Clear_Short],
    index=ordered_cols
)
ax = summary.plot.bar(color='lightgray', edgecolor='black', rot=0, title='Cost Reduction (Average)')
for p in ax.patches:
    reduction = (means.Complex - p.get_height()) / means.Complex * 100
    label = f"-{reduction:.1f}%" if reduction > 0.1 else "Base"
    ax.annotate(label, (p.get_x() + p.get_width() / 2, p.get_height()),
                ha='center', va='bottom', fontsize=10, fontweight='bold', color='black')
plt.tight_layout()
plt.savefig('2_cost_reduction_bw.png')
plt.close()

# --- ENVIRONMENTAL IMPACT ESTIMATE ---
ENERGY_PER_TOKEN_KWH = 0.0000040
PUE = 1.2
CARBON_INTENSITY_G_KWH = 250

co2_mg = means * ENERGY_PER_TOKEN_KWH * PUE * CARBON_INTENSITY_G_KWH * 1000
print("\n=== VALORI PER LA TABELLA LATEX ===")
print(f"{'Condition':<15} | {'Avg Tokens':<12} | {'CO2 (mg)':<10}")
print("-" * 45)
for col in ordered_cols:
    print(f"{col:<15} | {means[col]:>12.2f} | {co2_mg[col]:>10.2f}")

print("\n=== ENVIRONMENTAL IMPACT ESTIMATE (mg CO2 per query) ===")
print(f"{'Prompt':<20} | {'Standard GPU (A100)'}")
print("-" * 40)
for col in ordered_cols:
    print(f"{col:<20} | {co2_mg[col]:>10.2f} mg")
