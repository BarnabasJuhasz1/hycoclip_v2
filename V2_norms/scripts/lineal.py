import matplotlib.pyplot as plt

words = [
    "entity", "object", "organism", "thing", "person", 
    "living being", "company", "plant", "fruit", 
    "apple", "my uncle joe"
]

norms = [
    0.1310241, 0.13529123, 0.13736346, 0.13287708, 0.14495397,
    0.13895676, 0.14721209, 0.15014, 0.15702578, 0.23229438, 0.35401973
]

plt.figure(figsize=(12, 2))

plt.scatter(norms, [0]*len(norms), marker="o", color="black")

# set ticks exactly at norm positions
plt.xticks(norms, words, rotation=45, ha="right")

plt.yticks([])  
plt.xlabel("Norm value")
plt.title("Words placed along norm values")
plt.tight_layout()
save_path = "/leonardo/home/userexternal/bjuhasz0/fast/barnabas/project_hierarchies/HyCoCLIP/V2_norms/lines/lineal3"
#plt.savefig(save_path, bbox_inches="tight", dpi=300)
plt.savefig(save_path)
plt.show()

