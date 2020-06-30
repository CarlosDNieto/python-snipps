import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go

pd.options.plotting.backend = 'plotly'

students = ["Carlos", "Luis", "Marcela", "Jimena"]
grades = [8.8, 5.4, 9, 10]

df = pd.DataFrame({"students":students, "grades":grades})

print(df)

fig = df[['students','grades']].plot.bar(x='students', y='grades')
fig.show()

