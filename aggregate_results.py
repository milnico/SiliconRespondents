"""Aggregate the baseline/ablation psychometrics into one comparison table.

Runs the R evaluator (via 05_simulator_psychometrics.py's machinery) for each
(mode x test-set) silicon CSV, parses the printed metrics, and prints a table of
mean Tucker congruence, per-factor congruence, Phi error, and item-total profile
correlation. Modes {a, b, c, full} x test sets {IPIP held-out, BFI OOD}.
"""
import re
import subprocess

PY = "/home/nac/anaconda3/envs/llm_train/bin/python"
IPIP_HUMAN = "data/item_holdout.parquet"
BFI_HUMAN = "bfi_human_long.csv"

# (label, human_file, silicon_csv)
RUNS = [
    ("a   no-traits        | IPIP", IPIP_HUMAN, "silicon_a_ipip.csv"),
    ("b   persona-verbal    | IPIP", IPIP_HUMAN, "silicon_bverb_ipip.csv"),
    ("b   persona-numeric   | IPIP", IPIP_HUMAN, "silicon_bnum_ipip.csv"),
    ("c   prefix+LoRA(nogate)| IPIP", IPIP_HUMAN, "silicon_c_ipip.csv"),
    ("full prefix+gatedLoRA  | IPIP", IPIP_HUMAN, "silicon_t06.csv"),
    ("a   no-traits        |  BFI", BFI_HUMAN, "silicon_a_bfi.csv"),
    ("b   persona-verbal    |  BFI", BFI_HUMAN, "silicon_bverb_bfi.csv"),
    ("b   persona-numeric   |  BFI", BFI_HUMAN, "silicon_bnum_bfi.csv"),
    ("c   prefix+LoRA(nogate)|  BFI", BFI_HUMAN, "silicon_c_bfi.csv"),
    ("full prefix+gatedLoRA  |  BFI", BFI_HUMAN, "silicon_bfi_t10.csv"),
]


def run_eval(human, silicon):
    out = subprocess.run([PY, "05_simulator_psychometrics.py", "--human", human, "--silicon", silicon],
                         capture_output=True, text=True).stdout
    g = lambda pat: (re.search(pat, out).group(1) if re.search(pat, out) else None)
    mean = g(r"MEAN congruence\s*=\s*([0-9.]+)")
    per = g(r"per factor:\s*([0-9. ]+)")
    phi = g(r"Frobenius error\s*=\s*([0-9.]+)")
    itc = g(r"item-total profile corr.*=\s*([0-9.\-]+)")
    return mean, per, phi, itc


def main():
    print(f"{'mode | test':32s} {'congruence':>10} {'Phi err':>8} {'item-tot':>8}   per-factor")
    print("-" * 92)
    for label, human, silicon in RUNS:
        mean, per, phi, itc = run_eval(human, silicon)
        mean = f"{float(mean):.3f}" if mean else "  n/a "
        phi = f"{float(phi):.3f}" if phi else "  n/a"
        itc = f"{float(itc):.3f}" if itc else "  n/a"
        per = (per or "").strip()
        print(f"{label:32s} {mean:>10} {phi:>8} {itc:>8}   {per}")


if __name__ == "__main__":
    main()
