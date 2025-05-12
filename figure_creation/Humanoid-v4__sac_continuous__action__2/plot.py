from tbparse import SummaryReader
import seaborn as sns
import matplotlib.pyplot as plt

import os
from tbparse import SummaryReader
import numpy as np

def plot(steps=1000000, palette=None):
    print("plotting...")
    os.makedirs('fig_3', exist_ok = True)
    log_dir = os.path.join(".")
    df = SummaryReader(log_dir, pivot=True, extra_columns={'dir_name'}).scalars

    df = df[["step", "Test/return", "dir_name"]]
    df = df.assign(dir_name=df["dir_name"].apply(lambda s: s.split('/')[0]))
    df = df[df["step"] < 2500000]
    df.to_csv('output.csv', index=False)

    fig = plt.figure(figsize=(5,5.5))
    ax = plt.gca()
    sns.set_theme(style='whitegrid')
    plt.grid(color='lightgray')

    g = sns.lineplot(data=df, x='step', y='Test/return', hue='dir_name', palette=palette)
    g.set(ylim=(0, 7000))
    plt.legend([],[], frameon=False)

    plt.xlabel('')
    plt.ylabel('')
    plt.savefig('fig_3/Humanoid-v4.png')
    plt.close(fig)
    print("Finish plotting.")

def main(steps, palette=None):
    plot(steps=steps, palette=palette)

if __name__ == '__main__':

    steps = 2500000
    yticks = np.arange(0, 3500+500, 500)

    palette = ['xkcd:jade', 'xkcd:deep sky blue', 'xkcd:coral', 'xkcd:orange', 'xkcd:violet', 'xkcd:mauve']
    main(steps, palette=palette)