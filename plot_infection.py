import matplotlib.pyplot as plt
from plot_scores import getPassOne

inf_25_human, inf_25_human_p = getPassOne("eval/A2_infect/inf_25/humaneval_score.txt")
inf_25_mbpp, inf_25_mbpp_p = getPassOne("eval/A2_infect/inf_25/mbpp_score.txt")

inf_50_human, inf_50_human_p = getPassOne("eval/A2_infect/inf_50/humaneval_score.txt")
inf_50_mbpp, inf_50_mbpp_p = getPassOne("eval/A2_infect/inf_50/mbpp_score.txt")

inf_75_human, inf_75_human_p = getPassOne("eval/A2_infect/inf_75/humaneval_score.txt")
inf_75_mbpp, inf_75_mbpp_p = getPassOne("eval/A2_infect/inf_75/mbpp_score.txt")

inf_100_human, inf_100_human_p = getPassOne("results\\eval\\degraded\\humaneval_score.txt")
inf_100_mbpp, inf_100_mbpp_p = getPassOne("results\\eval\\degraded\\mbpp_score.txt")

human_base = [inf_25_human, inf_50_human, inf_75_human, inf_100_human]
human_plus = [inf_25_human_p, inf_50_human_p, inf_75_human_p, inf_100_human_p]
mbpp_base = [inf_25_mbpp, inf_50_mbpp, inf_75_mbpp, inf_100_mbpp]
mbpp_plus = [inf_25_mbpp_p, inf_50_mbpp_p, inf_75_mbpp_p, inf_100_mbpp_p]

precentages = [0.25, 0.5, 0.75, 1]

plt.plot(precentages, human_base, marker='o', color='r', label='HumanEval (base)')
plt.plot(precentages, human_plus, marker='s', linestyle='--', color='r', label='HumanEval+')
plt.plot(precentages, mbpp_base, marker='o', color='b', label='MBPP (base)')
plt.plot(precentages, mbpp_plus, marker='s', linestyle='--', color='b', label='MBPP+')

plt.xlabel("Precentage of data that is smell injected")
plt.ylabel("pass@1 score")
plt.title("Precentage of smell injected data and scores")
plt.grid(True)
plt.legend()

plt.show()