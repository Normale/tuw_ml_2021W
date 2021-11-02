import pandas as pd
import plotly.express as px

df = pd.read_csv("Datasets/winequality-white.csv", delimiter=';')

for x in df.columns.tolist():
  print(x)
pd.set_option('display.max_columns', None)
print("length overall", len(df))
print(df.describe())
print(df.shape)
#for name, values in df.iteritems():
 #   fig = px.histogram(df, x=name)
  #  fig.show()