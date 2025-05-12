import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
colors20 = cm.get_cmap("tab20").colors
datasets = ["pt-Dexnet-150-ft-realtrainset6", "pt-EGAD-150-ft-realtrainset6", "pt-GFDB-006-150-ft-realtrainset6", "pt-GFDB-030-150-ft-realtrainset6", "pt-GFDB-070-150-ft-realtrainset6", "pt-GFDB-006-030-070-150-ft-realtrainset6", "pt-Primitive-150-ft-realtrainset6", "realtrainset-6"]
window_size = 20
weights = np.ones(window_size) / window_size
step = 20
fig, ax = plt.subplots(figsize=(10,8))
i = 0
for dataset in datasets:
    model_path = f"/home/deepstation/grasp-fractal/gqcnn-sim/3rdparty/dexnet/deps/gqcnn/models/{dataset}"
    train_eval_iter = np.load(f"{model_path}/train_eval_iters.npy")
    train_error = np.load(f"{model_path}/train_errors.npy")
    #val_eval_iter = np.load(f"{model_path}/val_eval_iters.npy")
    #val_error = np.load(f"{model_path}/val_errors.npy")
    train_error2 = np.convolve(train_error, weights, mode='same')
    color = colors20[i % len(colors20)]
    ax.plot(train_eval_iter[::step], 100-train_error2[::step], color=color, label=dataset, linewidth=1, markersize=1)
    #ax.plot(val_eval_iter, 100-val_error, "o-b", label="val", linewidth=1, markersize=1)
    i = i +1 
ax.legend()
plt.show()