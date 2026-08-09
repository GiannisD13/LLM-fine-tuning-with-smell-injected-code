import matplotlib.pyplot as plt
import numpy as np
from plot_scores import getPassOne

clean_human, clean_human_p = getPassOne("results\\eval\\clean\\humaneval_score.txt")
clean_mbpp, clean_mbpp_p = getPassOne("results\\eval\\clean\\mbpp_score.txt")

degraded_human, degraded_human_p = getPassOne("results\\eval\\degraded\\humaneval_score.txt")
degraded_mbpp, degraded_mbpp_p = getPassOne("results\\eval\\degraded\\mbpp_score.txt")

smell_human, smell_human_p = getPassOne("eval/B_isol/smell/humaneval_score.txt")
smell_mbpp, smell_mbpp_p = getPassOne("eval/B_isol/smell/mbpp_score.txt")

bug_human, bug_human_p = getPassOne("eval/B_isol/bug/humaneval_score.txt")
bug_mbpp, bug_mbpp_p = getPassOne("eval/B_isol/bug/mbpp_score.txt")

models = ["clean", "smell-only", "bug-only", "original-degraded"]

series = {
    "HumanEval (base)": ((clean_human, smell_human, bug_human, degraded_human), '#4C72B0'),
    "HumanEval+":       ((clean_human_p, smell_human_p, bug_human_p, degraded_human_p), '#A3BEDC'),
    "MBPP (base)":      ((clean_mbpp, smell_mbpp, bug_mbpp, degraded_mbpp), '#DD8452'),
    "MBPP+":            ((clean_mbpp_p, smell_mbpp_p, bug_mbpp_p, degraded_mbpp_p), '#F0C0A0'),
}

x = np.arange(len(models))
width = 0.2

fig, ax = plt.subplots(figsize=(11, 6))
for i, (label, (values, color)) in enumerate(series.items()):
    offset = (i - 1.5) * width
    rects = ax.bar(x + offset, values, width, label=label, color=color)
    ax.bar_label(rects, fmt='%.2f', padding=3, fontsize=8)

ax.set_ylabel('Pass@1 Score', fontsize=12)
ax.set_xlabel('Model Variant', fontsize=12)
ax.set_title('Pass@1 Score Comparison Across Benchmarks', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend(title="Benchmark")
ax.grid(axis='y', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()