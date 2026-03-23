import numpy as np
from sklearn.metrics import confusion_matrix
import pandas as pd

data = np.load("models/validation_diagnostics.npz")

y_true = data["y_true"]
y_pred = data["y_pred"]

class_names = ["VBF", "ggF", "Background"]
labels = [0, 1, 2]

cm = confusion_matrix(y_true, y_pred, labels=labels)

rows = []

for i, name in enumerate(class_names):
    true_total = cm[i, :].sum()
    pred_total = cm[:, i].sum()

    efficiency = cm[i, i] / true_total if true_total > 0 else 0.0
    purity     = cm[i, i] / pred_total if pred_total > 0 else 0.0

    rows.append({
        "Class": name,
        "Efficiency": efficiency,
        "Purity": purity,
        "Events (true)": true_total,
        "Events (pred)": pred_total
    })

df = pd.DataFrame(rows)
print(df)
df.to_csv("models/efficiency_purity_summary.csv", index=False)

