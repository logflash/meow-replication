from tbparse import SummaryReader
import seaborn as sns
import matplotlib.pyplot as plt

import os
from tbparse import SummaryReader
import numpy as np

def plot(steps=1000000, yticks=None, xticks=None, palette=None):
    print("plotting...")
    os.makedirs('fig_3', exist_ok = True)
    log_dir = os.path.join(".")
    df = SummaryReader(log_dir, pivot=True, extra_columns={'dir_name'}).scalars

    df = df[["Steps", "Test/return", "dir_name"]]
    df = df.assign(dir_name=df["dir_name"].apply(lambda s: s.split('/')[0]))
    df.to_csv('output.csv', index=False)

    fig = plt.figure(figsize=(5,5.5))
    ax = plt.gca()
    sns.set_theme(style='whitegrid')
    plt.grid(color='lightgray')

    g = sns.lineplot(data=df, x='Steps', y='Test/return', hue='dir_name', palette=palette)
    g.set(xlim=(0, steps))
    g.set(ylim=(yticks[0], yticks[-1]))
    if xticks is not None:
        g.set_xticks(xticks)
    if yticks is not None:
        g.set_yticks(yticks)
    plt.legend([],[], frameon=False)

    plt.xlabel('')
    plt.ylabel('')
    plt.savefig('fig_3/Ant-v4.png')
    plt.close(fig)
    print("Finish plotting.")

def main(steps, yticks, xticks, palette=None):
    plot(steps=steps, yticks=yticks, xticks=xticks, palette=palette)

if __name__ == '__main__':

    steps = 4000000
    yticks = np.arange(-1500, 7500+1500, 1500)
    xticks = np.arange(0, steps+1, 1000000)

    palette = ['xkcd:jade', 'xkcd:deep sky blue', 'xkcd:coral', 'xkcd:orange', 'xkcd:violet', 'xkcd:mauve']
    main(steps, yticks, xticks, palette=palette)