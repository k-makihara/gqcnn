import numpy as np
import matplotlib.pyplot as plt

fname = "/home/deepstation/Downloads/adv_synth/depth_ims_tf_00000.npz"
f = np.load(fname)["arr_0"]
print(f[0].shape)
for i in range(len(f)):
    plt.subplot(10, 10, i % 100+1)
    plt.imshow(f[i].squeeze(-1), cmap="binary")
    plt.axis("off")
    if (i+1) % 100 == 0:
        plt.show()