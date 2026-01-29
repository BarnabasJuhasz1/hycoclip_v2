import matplotlib.pyplot as plt
import random

# Domains with 5 levels each
domain_words = {
    1: ["entity", "thing", "object", "laptop", "a silver MacBook Pro with stickers"],
    2: ["entity", "living being", "organism", "person", "my uncle Joe fixing his car"],
    3: ["entity", "living being", "organism", "plant", "a rose bush in full bloom"],
    4: ["entity", "organization", "company", "technology company", "Apple Inc. releasing a new iPhone"],
    5: ["entity", "living being", "organism", "fruit", "a shiny green apple on a wooden table"]
}

# Assign colors to each domain
domain_colors = {
    1: "blue",
    2: "green",
    3: "orange",
    4: "red",
    5: "purple"
}

plt.figure(figsize=(14, 6))

# We'll spread the words for each domain horizontally with an offset
x_base = 0
x_ticks = []
x_labels = []

for domain_id, words in domain_words.items():
    # Generate random norm values
    norms = [round(random.uniform(0.2, 0.35), 3) for _ in words]
    
    # Horizontal positions with small spacing for this domain
    x_positions = [x_base + i for i in range(len(words))]
    
    plt.plot(x_positions, norms, marker='o', linestyle='-', color=domain_colors[domain_id], label=f"Domain {domain_id}")
    
    # Record tick positions and labels
    x_ticks.extend(x_positions)
    x_labels.extend(words)
    
    # Update base for next domain (add extra space between domains)
    x_base += len(words) + 2

plt.xticks(x_ticks, x_labels, rotation=45, ha="right")
plt.ylabel("Norm value")
plt.title("Word Norms Across Domains")
plt.legend()
plt.tight_layout()

save_path = "/leonardo/home/userexternal/bjuhasz0/fast/barnabas/project_hierarchies/HyCoCLIP/V2_norms/lines/multi_domain_scatter2"
plt.savefig(save_path, bbox_inches="tight", dpi=300)
plt.show()
