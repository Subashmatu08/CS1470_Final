
import argparse
import pandas as pd
import matplotlib.pyplot as plt

def plot_ref_size(df):
    sub = df[df['ablation']=="ref_set_size"]

    
    print("REF SET SIZE DATA\n", sub)   

    x = sub['param'].astype(int)
    y = sub['AUC'] * 100
    plt.figure()
    plt.plot(x, y, marker='o')
    plt.xlabel("Reference Set Size")
    plt.ylabel("ROC-AUC (%)")
    plt.title("Fig. 3a – Reference-set size ablation")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("fig3a_ref_size.png")
    print("Saved fig3a_ref_size.png")

def plot_lambda(df):
    sub = df[df['ablation']=='lambda_percentile']
    x   = sub['param'].astype(int)
    auc = sub['AUC']
    ap  = sub['AP']

    plt.figure()
    plt.plot(x, ap , label='AP', marker='o')
    plt.plot(x, auc, label='ROC-AUC', marker='o')
    plt.xlabel("Percentile λ")
    plt.ylabel("FV-AVG Performance (%)")
    plt.xlim(0,100)
    plt.legend()
    plt.grid(True)
    plt.savefig("fig3b_lambda.png", dpi=300)


def plot_encoder_compare(df):
    sub = df[df['ablation']=="encoder_compare"]
    encs = sub['param']
    fake_pct = sub['fake_higher'] * 100
    real_pct = sub['real_higher'] * 100

    x = range(len(encs))
    width = 0.35
    plt.figure()
    plt.bar(x, fake_pct, width, label="Fake>Real")
    plt.bar([i+width for i in x], real_pct, width, label="Real>Fake")
    plt.xlabel("Encoder")
    plt.ylabel("Percentage of pairs")
    plt.title("Fig. 3c – Encoder truth-score comparison")
    plt.xticks([i+width/2 for i in x], encs)
    plt.legend()
    plt.tight_layout()
    plt.savefig("fig3c_encoder_compare.png")
    print("Saved fig3c_encoder_compare.png")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True,
                        help="Path to your ablation_results.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    print("=== HEAD ===")
    print(df.head(), "\n")
    print("=== COLS ===", df.columns.tolist())
    print("=== UNIQUE ABLATIONS ===", df['ablation'].unique())
    plot_ref_size(df)
    plot_lambda(df)
    plot_encoder_compare(df)

if __name__ == "__main__":
    main()
