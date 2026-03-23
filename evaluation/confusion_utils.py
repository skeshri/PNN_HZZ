import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def plot_confusion(
    y_true,
    y_pred,
    class_names,
    title,
    outname,
    normalize="true",
):
    """
    Plot confusion matrix with both counts and normalized values.
    """

    # -------------------------------------------------
    # Safety checks
    # -------------------------------------------------
    if len(y_true) == 0 or len(y_pred) == 0:
        print(f"[plot_confusion] WARNING: empty input for '{title}', skipping.")
        return None, None

    labels = list(range(len(class_names)))

    cm_counts = confusion_matrix(
        y_true, y_pred, labels=labels
    )

    cm_norm = confusion_matrix(
        y_true, y_pred, labels=labels, normalize=normalize
    )

    # -------------------------------------------------
    # Plot
    # -------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)

    # Colorbar label
    if normalize == "true":
        cbar_label = "Fraction of true class"
    elif normalize == "pred":
        cbar_label = "Fraction of predicted class"
    elif normalize == "all":
        cbar_label = "Fraction of all events"
    else:
        cbar_label = "Counts (unnormalized)"

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(cbar_label)

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)

    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)

    # -------------------------------------------------
    # Annotate cells
    # -------------------------------------------------
    for i in range(cm_counts.shape[0]):
        for j in range(cm_counts.shape[1]):
            count = cm_counts[i, j]
            frac  = cm_norm[i, j]

            text_color = "white" if frac > 0.6 else "black"

            ax.text(
                j, i,
                f"{count}\n({frac:.2f})",
                ha="center",
                va="center",
                color=text_color,
                fontsize=10
            )

    # -------------------------------------------------
    # SAVE + CLEANUP  ✅ THIS WAS MISSING
    # -------------------------------------------------
    plt.tight_layout()
    plt.savefig(outname)
    plt.close(fig)

    print(f"[plot_confusion] Saved {outname}")

    return cm_counts, cm_norm

