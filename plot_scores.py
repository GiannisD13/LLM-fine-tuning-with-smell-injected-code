import sys
import matplotlib.pyplot as plt
import re

# Τα warnings είναι ελληνικά — το default cp1252 stdout στα Windows σκάειι
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

"""

ΔΕΝ ΔΙΑΒΑΖΕΙ ΤΑ ΤΕΣΤ + ΚΑΝΕΙ ΠΛΟΤ ΜΟΝΟ ΤΑ ΑΠΛΑ 

"""

def getPassOne(filepath: str):
    # Το evalplus γράφει δύο γραμμές pass@1: πρώτη = base, δεύτερη = plus
    # (humaneval+ / mbpp+, αυστηρότερα test). Επιστρέφει (base, plus).
    try:
        with open(filepath, "r", encoding="utf-8") as file:
            matches = re.findall(r"pass@1:\s*([\d.]+)", file.read())
    except FileNotFoundError:
        print(f"προσοχή: λείπει {filepath}")
        return None, None
    base = float(matches[0]) if len(matches) >= 1 else None
    plus = float(matches[1]) if len(matches) >= 2 else None
    if base is None:
        print(f"προσοχή: δεν βρέθηκε pass@1 στο {filepath}")
    return base, plus


if __name__ == "__main__":
    deg_25_human, deg_25_human_p = getPassOne("eval/A_size/deg_25/humaneval_score.txt")
    deg_25_mbpp, deg_25_mbpp_p = getPassOne("eval/A_size/deg_25/mbpp_score.txt")

    deg_50_human, deg_50_human_p = getPassOne("eval/A_size/deg_50/humaneval_score.txt")
    deg_50_mbpp, deg_50_mbpp_p = getPassOne("eval/A_size/deg_50/mbpp_score.txt")

    deg_75_human, deg_75_human_p = getPassOne("eval/A_size/deg_75/humaneval_score.txt")
    deg_75_mbpp, deg_75_mbpp_p = getPassOne("eval/A_size/deg_75/mbpp_score.txt")

    deg_100_human, deg_100_human_p = getPassOne("results\\eval\\degraded\\humaneval_score.txt")
    deg_100_mbpp, deg_100_mbpp_p = getPassOne("results\\eval\\degraded\\mbpp_score.txt")

    human_base = [deg_25_human, deg_50_human, deg_75_human, deg_100_human]
    human_plus = [deg_25_human_p, deg_50_human_p, deg_75_human_p, deg_100_human_p]
    mbpp_base = [deg_25_mbpp, deg_50_mbpp, deg_75_mbpp, deg_100_mbpp]
    mbpp_plus = [deg_25_mbpp_p, deg_50_mbpp_p, deg_75_mbpp_p, deg_100_mbpp_p]

    precentages = [0.25, 0.5, 0.75, 1]

    plt.plot(precentages, human_base, marker='o', color='r', label='HumanEval (base)')
    plt.plot(precentages, human_plus, marker='s', linestyle='--', color='r', label='HumanEval+')
    plt.plot(precentages, mbpp_base, marker='o', color='b', label='MBPP (base)')
    plt.plot(precentages, mbpp_plus, marker='s', linestyle='--', color='b', label='MBPP+')

    plt.xlabel("Precentage of data of the total corpus used")
    plt.ylabel("pass@1 score")
    plt.title("Capacity of data and model performance")
    plt.grid(True)
    plt.legend()

    plt.show()




