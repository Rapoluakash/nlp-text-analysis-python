from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Sample text to generate WordCloud
text = (
    "Python Python Python Matplotlib Matplotlib Seaborn Network Plot Violin "
    "Chart Pandas Datascience Wordcloud Spider Radar Parallel Alpha Color "
    "Brewer Density Scatter Barplot Boxplot Violinplot Treemap Stacked Area "
    "Chart Visualization Dataviz Donut Pie Time-Series Wordcloud Wordcloud Sankey Bubble"
)

# WordCloud Configuration
wordcloud = WordCloud(
    width=420,
    height=200,
    margin=2,
    background_color='black',
    colormap='Accent',
    mode='RGBA'
).generate(text)

# Plot the WordCloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()
