import numpy as np
import glob
import statistics

#dataset_list = ["Dexnet-150", "EGAD-150", "GFDB-006-150", "GFDB-030-150", "GFDB-070-150", "GFDB-006-030-070-150", "Primitive-150"]
#dataset_list = ["Dexnet-150", "EGAD-150", "Grasp-FractalMeshDB-006-150-v2", "Grasp-FractalMeshDB-030-150-v2", "Grasp-FractalMeshDB-070-150-v2", "Grasp-FractalMeshDB-006-030-070-150", "Primitive-150", "realtrainset-6"]
#dataset_list = ["GFDB-006-150-x10", "GFDB-030-150-x10", "GFDB-070-150-x10", "GFDB-006-030-070-150-full"]
dataset_list = ["pt-Dexnet-10-ft-realtrainset", "pt-Dexnet-50-ft-realtrainset", "pt-Dexnet-150-ft-realtrainset", "pt-EGAD-10-ft-realtrainset", "pt-EGAD-50-ft-realtrainset", "pt-EGAD-150-ft-realtrainset", "pt-GFDB-006-10-ft-realtrainset", "pt-GFDB-006-50-ft-realtrainset", "pt-GFDB-006-150-ft-realtrainset", "realtrainset"]



for dataset in dataset_list:
    #print(dataset)
    res_mean = []
    res_variance = []
    #for i in range(9):
    for i in range(1):
        res_each = []
        #for j in range(5):
        for j in range(2):
            #DATASET_NAME = f"{dataset}-{int((i+1)*10):02}-seed{j+1}"
            DATASET_NAME = f"{dataset}"
            #print(DATASET_NAME)
            try:
                train_gt = np.load(f"/home/deepstation/grasp-fractal/gqcnn-sim/3rdparty/dexnet/deps/gqcnn/analysis_real/{DATASET_NAME}/train_result.cres/labels.npz")["arr_0"]
                val_gt = np.load(f"/home/deepstation/grasp-fractal/gqcnn-sim/3rdparty/dexnet/deps/gqcnn/analysis_real/{DATASET_NAME}/val_result.cres/labels.npz")["arr_0"]
                train_pred = np.load(f"/home/deepstation/grasp-fractal/gqcnn-sim/3rdparty/dexnet/deps/gqcnn/analysis_real/{DATASET_NAME}/train_result.cres/predictions.npz")["arr_0"]
                val_pred = np.load(f"/home/deepstation/grasp-fractal/gqcnn-sim/3rdparty/dexnet/deps/gqcnn/analysis_real/{DATASET_NAME}/val_result.cres/predictions.npz")["arr_0"]


                pred_probs = np.concatenate([train_pred, val_pred], axis=0)
                true_labels = np.concatenate([train_gt, val_gt], axis=0)
                #print(train_gt)
                #print(train_pred)
                #print(val_gt)
                #print(val_pred)

                #print(len(pred_probs))
                #print(len(true_labels))

                pred_labels = (pred_probs >= 0.5).astype(int)

                # 正しく予測できた数をカウント
                correct = np.sum(pred_labels == true_labels)

                # 全サンプル数
                total = true_labels.shape[0]

                # 分類精度
                accuracy = correct / total

                #print(f"Accuracy: {accuracy:.4f} ({correct}/{total})")
                res_each.append(accuracy)
            except:
                continue
        mean_each = statistics.mean(res_each)
        variance_each = statistics.variance(res_each)
        res_mean.append(mean_each)
        res_variance.append(variance_each)

    print(dataset)
    print(res_mean)
    print(res_variance)

