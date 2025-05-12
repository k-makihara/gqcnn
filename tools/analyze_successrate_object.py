import numpy as np
import glob
import json
import copy
from collections import defaultdict


#dataset_list = ["Dexnet-150", "EGAD-150", "GFDB-006-150", "GFDB-030-150", "GFDB-070-150", "GFDB-006-030-070-150", "Primitive-150"]
#dataset_list = ["Dexnet-150", "EGAD-150", "Grasp-FractalMeshDB-006-150-v2", "Grasp-FractalMeshDB-030-150-v2", "Grasp-FractalMeshDB-070-150-v2", "Grasp-FractalMeshDB-006-030-070-150", "Primitive-150", "realtrainset-6", "realtrainset-6-ep500"]
#dataset_list = ["Dexnet-150", "EGAD-150", "Grasp-FractalMeshDB-006-150-v2", "Grasp-FractalMeshDB-030-150-v2", "Grasp-FractalMeshDB-070-150-v2", "Grasp-FractalMeshDB-006-030-070-150", "Primitive-150", "realtrainset"]
#dataset_list = ["pt-Dexnet-150-ft-realtrainset", "pt-EGAD-150-ft-realtrainset", "pt-GFDB-006-150-ft-realtrainset", "pt-GFDB-030-150-ft-realtrainset", "pt-GFDB-070-150-ft-realtrainset", "pt-GFDB-006-030-070-150-ft-realtrainset", "pt-Primitive-150-ft-realtrainset", "realtrainset"]
#dataset_list = ["Dexnet-EGAD-150", "Dexnet-GFDB006-150", "Dexnet-GFDB030-150", "Dexnet-GFDB070-150", "Dexnet-GFDB006030070-150", "Dexnet-GFDB006030070full-150", "Dexnet-Primitive-150"]
dataset_list = ["pt-Dexnet-EGAD-150-ft-realtrainset", "pt-Dexnet-GFDB006-150-ft-realtrainset", "pt-Dexnet-GFDB030-150-ft-realtrainset", "pt-Dexnet-GFDB070-150-ft-realtrainset", "pt-Dexnet-GFDB006030070-150-ft-realtrainset", "pt-Dexnet-GFDB006030070full-150-ft-realtrainset", "pt-Primitive-150-ft-realtrainset", "realtrainset"]
#dataset_list = ["Dexnet-150"]



for dataset in dataset_list:
    res = []
    obj_list = []
    success_rate_list = []
    for i in range(1):
        #DATASET_NAME = f"{dataset}-{int((i+1)*10):02}"
        DATASET_NAME = f"{dataset}"
        print(DATASET_NAME)
        try:
            train_gt = np.load(f"/home/deepstation/grasp-fractal/gqcnn-sim/3rdparty/dexnet/deps/gqcnn/analysis_real/{DATASET_NAME}/train_result.cres/labels.npz")["arr_0"]
            val_gt = np.load(f"/home/deepstation/grasp-fractal/gqcnn-sim/3rdparty/dexnet/deps/gqcnn/analysis_real/{DATASET_NAME}/val_result.cres/labels.npz")["arr_0"]
            train_pred = np.load(f"/home/deepstation/grasp-fractal/gqcnn-sim/3rdparty/dexnet/deps/gqcnn/analysis_real/{DATASET_NAME}/train_result.cres/predictions.npz")["arr_0"]
            val_pred = np.load(f"/home/deepstation/grasp-fractal/gqcnn-sim/3rdparty/dexnet/deps/gqcnn/analysis_real/{DATASET_NAME}/val_result.cres/predictions.npz")["arr_0"]

            train_indices = np.load("/home/deepstation/Downloads/realtrainset/splits/image_wise/train_indices.npz")["arr_0"]
            val_indices = np.load("/home/deepstation/Downloads/realtrainset/splits/image_wise/val_indices.npz")["arr_0"]
            indices = np.concatenate([train_indices, val_indices], axis=0)

    

            obj_labels_list = sorted(glob.glob("/home/deepstation/Downloads/realtrainset/tensors/obj_labels_*.npz"))
            obj_labels_all = None
            for obj_labels_file in obj_labels_list:
                obj_labels = np.load(obj_labels_file)["arr_0"]
                if obj_labels_all is None:
                    obj_labels_all = copy.copy(obj_labels)
                else:
                    obj_labels_all = np.concatenate([obj_labels_all, obj_labels], axis=0)
            obj_labels_pred = obj_labels_all[indices]
            
            

            obj_names = json.load(open("/home/deepstation/Downloads/realtrainset/object_category_map.json","rb"))
            obj_ids = {v: k for k, v in obj_names.items()}


            pred_probs = np.concatenate([train_pred, val_pred], axis=0)
            true_labels = np.concatenate([train_gt, val_gt], axis=0)
            #print(train_gt)
            #print(train_pred)
            #print(val_gt)
            #print(val_pred)

            #print(len(pred_probs))
            #print(len(true_labels))

            pred_labels = (pred_probs >= 0.5).astype(int)

            total_counts   = defaultdict(int)
            correct_counts = defaultdict(int)

            for t, p, y in zip(obj_labels_pred, pred_labels, true_labels):
                total_counts[t] += 1
                if p == y:
                    correct_counts[t] += 1

            #print(dict(correct_counts))

            # 成功率を計算
            success_rate = {}
            for t in total_counts:
                success_rate[obj_ids[t]] = correct_counts[t] / total_counts[t]
                obj_list.append(obj_ids[t])
                success_rate_list.append(correct_counts[t] / total_counts[t])

            #print(success_rate)
            #print(obj_list)
            print(success_rate_list)

            #print(f"Accuracy: {accuracy:.4f} ({correct}/{total})")
            #res.append(accuracy)
        except:
            continue
    #print(dataset)
    #print(res)

