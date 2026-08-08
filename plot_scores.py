import matplotlib.pyplot as plt
import re

"""

ΔΕΝ ΔΙΑΒΑΖΕΙ ΤΑ ΤΕΣΤ + ΚΑΝΕΙ ΠΛΟΤ ΜΟΝΟ ΤΑ ΑΠΛΑ 

"""

def getPassOne(filepath: str): 
    with open(filepath, "r", encoding="utf-8") as file:
        passone = re.search(r"pass@1:\s*([\d.]+)", file.read())
    if passone:
        print(f"Found it: {passone.group(1)}")
        return float(passone.group(1))

deg_25_human = getPassOne("eval/A_size/deg_25/humaneval_score.txt")
deg_25_mbpp = getPassOne("eval/A_size/deg_25/mbpp_score.txt")

deg_50_human = getPassOne("eval/A_size/deg_50/humaneval_score.txt")
deg_50_mbpp = getPassOne("eval/A_size/deg_50/mbpp_score.txt")

deg_75_human = getPassOne("eval/A_size/deg_75/humaneval_score.txt")
deg_75_mbpp = getPassOne("eval/A_size/deg_75/mbpp_score.txt")

deg_100_human = getPassOne("results\\eval\\degraded\\humaneval_score.txt")
deg_100_mbpp = getPassOne("results\\eval\\degraded\\mbpp_score.txt")

human_scores = [deg_25_human, deg_50_human, deg_75_human, deg_100_human]
mbpp_scores = [deg_25_mbpp, deg_50_mbpp, deg_75_mbpp, deg_100_mbpp]

precentages = [0.25 , 0.5, 0.75, 1]

plt.plot(precentages, human_scores, marker='o', color='r', label='HumanEval Scores')
plt.plot(precentages, mbpp_scores, marker='o', color='b', label='MBPP Scores')

plt.xlabel("Precentage of data of the total corpus used")
plt.ylabel("pass@1 score")
plt.title("Capacity of data and model performance")
plt.grid(True)
plt.legend()

plt.show()




