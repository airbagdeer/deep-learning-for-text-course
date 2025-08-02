import pandas as pd
import matplotlib.pyplot as plt

def load_and_downsample(path, max_points=500):
    df = pd.read_csv(path, header=None, names=["samples_number", "acrr"])
    df["samples_number"] = df["samples_number"] / 100
    if len(df) > max_points:
        df = df.iloc[::len(df)//max_points]
    return df

def plot_csvs(csv_paths, labels, output_path, max_points=500):
    plt.figure(figsize=(12, 6))
    
    for path, label in zip(csv_paths, labels):
        df = load_and_downsample(path, max_points)
        plt.plot(df["samples_number"], df["acrr"], label=label)

    plt.xlabel("samples number / 100")
    plt.ylabel("Dev Accuracy")
    plt.title("Dev Accuracy For Pos Tagging Over Different Word Representations")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


if(__name__ == '__main__'):
    # Example usage:
    csv_paths = ["/Users/itaygradenwits/Documents/biu/deep-nlp/deep-learning-for-text-course/assignment3/q3/logs/logs-pos-a.csv",
                 "/Users/itaygradenwits/Documents/biu/deep-nlp/deep-learning-for-text-course/assignment3/q3/logs/logs-pos-b.csv",
                 "/Users/itaygradenwits/Documents/biu/deep-nlp/deep-learning-for-text-course/assignment3/q3/logs/logs-pos-c.csv",
                 "/Users/itaygradenwits/Documents/biu/deep-nlp/deep-learning-for-text-course/assignment3/q3/logs/logs-pos-d.csv"
                ]
    labels = ["A", "B", "C", "D"]
    output_path = "/Users/itaygradenwits/Documents/biu/deep-nlp/deep-learning-for-text-course/assignment3/q3/images/pos.png"

    plot_csvs(csv_paths, labels, output_path)