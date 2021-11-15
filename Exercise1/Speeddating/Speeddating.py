import pandas as pd
import plotly.express as px

df = pd.read_csv("../Datasets/speeddating_1.csv", encoding="ISO-8859-1")

values = df['d_expected_happy_with_sd_people'].value_counts()

print(values)