import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('QtAgg')
import seaborn as sns
import pandas as pd
import numpy as np

### Global settings and vars ###
plt.style.use('seaborn-v0_8-bright')
INPUT_CSV = "./output/csv/eval.csv"

def score_histogram_plot(outfile):
    try:
        df = pd.read_csv(INPUT_CSV)
    except FileNotFoundError:
        print("ERROR: File not found.")
        return

    # Define bins.
    bins = np.arange(0, 99)

    mu = sum(df['score']) / len(df['score'])
    print(mu)

    # Create the histogram and capture the 'patches'.
    #n, bins, patches = plt.hist(df['score'], bins=bins, density=True, edgecolor='black', alpha=1, align='left')
    sns.histplot(
        data=df, 
        x='score', 
        bins=bins, 
        color="purple",
        stat='density', 
        kde=True, 
        edgecolor='black', 
        discrete=False,
        label=f"$\\mu={mu:.1f}$"
    )

    # Formatting.
    plt.xlabel('Score', fontsize=12)
    plt.ylabel('Probability', fontsize=12)
    
    plt.xticks(np.arange(0, 100, 5))
    plt.xlim(-1, 98)
    plt.grid(axis='y', linestyle='--', alpha=0.4)
    plt.legend()
    
    # 6. Save and show
    plt.tight_layout()
    plt.savefig(outfile)
    plt.close()


if __name__ == '__main__':
    score_histogram_plot("./output/plots/score_histogram_plot.png")

