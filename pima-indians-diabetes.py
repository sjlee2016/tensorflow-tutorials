import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns 
df = pd.read_csv('./dataset/pima-indians-diabetes.csv',
names = ["pregnant", "plasma", "pressure", "thicvkness", "insulin", "BMI", "pedigree", "age", "class"])
print(df[['pregnant', 'class']].groupby(['pregnant'], as_index=False).mean().sort_values(by='pregnant', ascending=True))
plt.figure(figsize=(10,10))
sns.heatmap(df.corr(), linewidths=0.1, vmax=0.5, cmap=plt.cm.gist_heat, linecolor='white', annot=True)
grid = sns.FacetGrid(df,col='class')
grid.map(plt.hist, 'plasma', bins=10)
plt.show()