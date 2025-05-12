import numpy as np
import glob

files = glob.glob("/home/deepstation/Downloads/gfdb_006_030_070_150/tensors/robust_ferrari_canny_*.npz")
print(len(files))

thresh = 0.002
count = 0
neg_count = 0
for file in files:
    f = np.load(file)["arr_0"]
    c = sum(1 for x in f if x <= thresh)
    neg_count = neg_count + c
    count = count + len(f)

print(neg_count)
print(count)

print(neg_count/count * 100)