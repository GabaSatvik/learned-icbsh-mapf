import pandas as pd

df = pd.read_csv('data/rollout_training_regimeB.csv')
print('=== Rollout Data Summary ===')
print('Total rows  :', len(df))
qids = df['query_id'].nunique()
print('Queries     :', qids)
print('Avg cand/q  :', round(len(df)/qids, 1))

print()
print('--- Rollout cost distribution ---')
print(df['rollout_cost'].describe().round(1))

print()
print('--- Tied vs un-tied queries ---')
variance = df.groupby('query_id')['rollout_cost'].nunique()
tied = (variance == 1).sum()
untied = (variance > 1).sum()
print('Tied   (no signal):', tied, '/', len(variance), f'= {100*tied/len(variance):.1f}%')
print('Untied (has signal):', untied, '/', len(variance), f'= {100*untied/len(variance):.1f}%')

print()
print('--- By map and k ---')
g = df.groupby(['map', 'k'])['query_id'].nunique().reset_index()
g.columns = ['map', 'k', 'queries']
print(g.to_string(index=False))

print()
PENALTY = 1500 * 1.5
print(f'Penalty value (B*1.5): {PENALTY}')
penalty_rows = (df['rollout_cost'] == PENALTY).sum()
print(f'Rows at penalty cost : {penalty_rows} / {len(df)} = {100*penalty_rows/len(df):.1f}%')

print()
print('--- Rollout cost counts (unique values) ---')
print(df['rollout_cost'].value_counts().head(10))
