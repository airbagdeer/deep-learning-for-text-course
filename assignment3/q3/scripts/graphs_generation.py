import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_downsample(path, max_points=500):
    df = pd.read_csv(path, header=None, names=["samples_number", "acrr"])
    df["samples_number"] = df["samples_number"] / 100
    if len(df) > max_points:
        df = df.iloc[::len(df)//max_points]
    return df

def plot_csvs(csv_paths, labels, output_path, max_points=500, transparent=False):
    sns.set(style="darkgrid")  # Use a nice modern theme
    plt.figure(figsize=(14, 7))

    markers = ['o', 's', '^', 'D']  # Circle, square, triangle, diamond
    palette = sns.color_palette("husl", len(csv_paths))  # Distinct colors

    for i, (path, label) in enumerate(zip(csv_paths, labels)):
        df = load_and_downsample(path, max_points)
        plt.plot(
            df["samples_number"],
            df["acrr"],
            label=label,
            linewidth=2.0,
            marker=markers[i % len(markers)],
            markersize=5,
            color=palette[i]
        )

    plt.xlabel("Samples Number / 100", fontsize=14)
    plt.ylabel("ACRR", fontsize=14)
    plt.title("ACRR vs. Samples Number", fontsize=16, weight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, transparent=transparent)
    plt.close()

if(__name__ == '__main__'):
    # Example usage:
    csv_paths = ["/Users/itaygradenwits/Documents/biu/deep-nlp/deep-learning-for-text-course/assignment3/q3/logs/loggs-pos-a.csv",
                 "/Users/itaygradenwits/Documents/biu/deep-nlp/deep-learning-for-text-course/assignment3/q3/logs/loggs-pos-a.csv",
                 "/Users/itaygradenwits/Documents/biu/deep-nlp/deep-learning-for-text-course/assignment3/q3/logs/loggs-pos-a.csv",
                 "/Users/itaygradenwits/Documents/biu/deep-nlp/deep-learning-for-text-course/assignment3/q3/logs/loggs-pos-a.csv"
                ]
    labels = ["A", "B", "C", "D"]
    output_path = "/Users/itaygradenwits/Documents/biu/deep-nlp/deep-learning-for-text-course/assignment3/q3/images/pos.png"

    plot_csvs(csv_paths, labels, output_path)