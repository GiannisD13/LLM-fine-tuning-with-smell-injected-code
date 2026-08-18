import sys
import matplotlib.pyplot as plt
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


"""

 Συμπλήρωσε εδώ: (issues, ncloc) ανά μοντέλο.

"""
SONAR = {
    "clean":             {"issues": 123, "ncloc": 2673},
    "smell-only":        {"issues": 3923, "ncloc": 5531},
    "bug-only":          {"issues": 63, "ncloc": 2999},
    "original-degraded": {"issues": 4122, "ncloc": 6319},
}


def issues_per_kloc(entry: dict) -> float:
    # issues ανά 1000 γραμμές κώδικα — κανονικοποιεί για άνισο μέγεθος output
    if entry["ncloc"] == 0:
        print(f"προσοχή: ncloc=0, δεν υπάρχουν δεδομένα SonarQube ακόμα")
        return 0.0
    return entry["issues"] / (entry["ncloc"] / 1000)


models = list(SONAR.keys())
density = [issues_per_kloc(SONAR[m]) for m in models]

x = np.arange(len(models))

fig, ax = plt.subplots(figsize=(10, 6))
rects = ax.bar(x, density, width=0.6, color='#C44E52')

ax.set_ylabel('Issues / KLOC', fontsize=12)
ax.set_xlabel('Model Variant', fontsize=12)
ax.set_title('Static Code Quality of Generations (SonarQube)', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.grid(axis='y', linestyle='--', alpha=0.5)
ax.bar_label(rects, fmt='%.1f', padding=3)

plt.tight_layout()
plt.show()
