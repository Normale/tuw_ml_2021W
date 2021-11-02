import pandas as pd
import plotly.express as px

df = pd.read_csv("speeddating_1.csv", encoding="ISO-8859-1")
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)

# for x in df.columns.tolist():
#   print(x)
print(len( df.columns.tolist()))
print(f"ROWS INCLUDING NULLS: {len(df[df['has_null'] == 1])}")
print("length overall", len(df))

# print(df['d_funny_important'].iloc(3)
# dummy = df['met'] == 1
# print(df.loc[(df['match'] >= 1) & (df['decision'] == 0)])
# print(df.iloc[:,3].unique())
df_2 = df[["match", "decision", "decision_o"]]
# print(df_2)
match_values = df['match'].value_counts()

df_3 = pd.DataFrame(match_values).reset_index()
df_3.columns = ['match', 'count']


#d_pref_o_funny
#['[16-20]' '[21-100]' '[0-15]']

fig = px.pie(df_3, values='count',names='match', title='% of matches', labels= ['Yes','No'])
fig.show()
