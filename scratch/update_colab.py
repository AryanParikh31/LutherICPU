import json

file_path = "LutherICPU_Colab_Trainer.ipynb"
with open(file_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Update cell 1 where git clone is performed
for cell in nb.get("cells", []):
    if cell.get("cell_type") == "code":
        src = "".join(cell.get("source", []))
        if "!git clone https://github.com/AryanParikh31/LutherICPU.git" in src:
            new_lines = []
            for line in cell["source"]:
                if "!git clone https://github.com/AryanParikh31/LutherICPU.git /content/LutherICPU" in line:
                    new_lines.append("BRANCH = \"LutherICPUv3\"\n")
                    new_lines.append("!git clone -b {BRANCH} https://github.com/AryanParikh31/LutherICPU.git /content/LutherICPU\n")
                elif "[SUCCESS] Native LutherICPU GPU environment initialized successfully!" in line:
                    new_lines.append("print(f\"\\n[SUCCESS] Native LutherICPU GPU environment initialized from branch '{BRANCH}' successfully!\")")
                else:
                    new_lines.append(line)
            cell["source"] = new_lines
            break

with open(file_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("Updated LutherICPU_Colab_Trainer.ipynb successfully!")
