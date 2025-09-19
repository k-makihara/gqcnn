import numpy as np
import glob
import json
import copy
from collections import defaultdict
from sklearn.metrics import precision_score, accuracy_score, f1_score, average_precision_score

#dataset_list = ["Dexnet-150", "EGAD-150", "Grasp-FractalMeshDB-006-150-v2", "Grasp-FractalMeshDB-030-150-v2", "Grasp-FractalMeshDB-070-150-v2", "Grasp-FractalMeshDB-006-030-070-150", "Primitive-150","realtrainset-6"]
#dataset_list = ["pt-Dexnet-150-ft-realtrainset6", "pt-EGAD-150-ft-realtrainset6", "pt-GFDB-006-150-ft-realtrainset6", "pt-GFDB-030-150-ft-realtrainset6", "pt-GFDB-070-150-ft-realtrainset6", "pt-GFDB-006-030-070-150-ft-realtrainset6", "pt-GFDB-006-030-070-150-full-ft-realtrainset6", "pt-Primitive-150-ft-realtrainset6", "realtrainset-6", "realtrainset-6-ep500"]
#dataset_list = ["Dexnet-150", "Dexnet-EGAD-150", "Dexnet-GFDB006-150", "Dexnet-GFDB030-150", "Dexnet-GFDB070-150", "Dexnet-GFDB006030070-150", "Dexnet-GFDB006030070full-150", "Dexnet-Primitive-150"]
#dataset_list = ["pt-Dexnet-EGAD-150-ft-realtrainset6", "pt-Dexnet-GFDB006-150-ft-realtrainset6", "pt-Dexnet-GFDB030-150-ft-realtrainset6", "pt-Dexnet-GFDB070-150-ft-realtrainset6", "pt-Dexnet-GFDB006030070-150-ft-realtrainset6", "pt-Dexnet-GFDB006030070full-150-ft-realtrainset6", "pt-Primitive-150-ft-realtrainset6", "realtrainset6"]
#dataset_list = ["Dexnet-150", "EGAD-150", "Grasp-FractalMeshDB-006-150-v2", "Grasp-FractalMeshDB-030-150-v2", "Grasp-FractalMeshDB-070-150-v2", "Grasp-FractalMeshDB-006-030-070-150", "Primitive-150", "realtrainset"]
#dataset_list = ["Dexnet-150"]
#dataset_list = ["pt-Dexnet-10-ft-realtrainset6", "pt-Dexnet-50-ft-realtrainset6", "pt-Dexnet-150-ft-realtrainset6", "pt-EGAD-10-ft-realtrainset6", "pt-EGAD-50-ft-realtrainset6", "pt-EGAD-150-ft-realtrainset6", "pt-GFDB-006-10-ft-realtrainset6", "pt-GFDB-006-50-ft-realtrainset6", "pt-GFDB-006-150-ft-realtrainset6", "realtrainset-6"]
#dataset_list = ["Dexnet-150-pos010", "Dexnet-150-pos020", "Dexnet-150-pos030", "Dexnet-150-pos040", "Dexnet-150-pos050", "Dexnet-150-pos060", "Dexnet-150-pos070", "Dexnet-150-pos080", "Dexnet-150-pos090"]
#dataset_list = ["Downloads_image_wise_trial_13_2025-09-10_13:47:45", "Downloads_image_wise_trial_12_2025-09-10_21:31:28"]
#dataset_list = ["Dexnet-best","EGAD-best","GFDB-006-best","GFDB-030-best","GFDB-070-best","GFDB-combined-best"]
#dataset_list = ["Primitive-best"]
dataset_list = ["Dexnet-fiximage-posneg","Primitive-fiximage-posneg","EGAD-fiximage-posneg","GFDB-006-fiximage-posneg","GFDB-030-fiximage-posneg","GFDB-070-fiximage-posneg"]


for dataset in dataset_list:
    
    
    overall_accuracy_list = []
    overall_pricision_list = []
    overall_macro_accuracy_list = []
    overall_macro_pricision_list = []
    overall_f1_list = []
    overall_ap_list = []
    for i in range(3):
        res = []
        obj_list = []
        success_rate_list = []
        pricision_list = []
        if i == 0:
            DATASET_NAME = f"{dataset}"
        else:
            DATASET_NAME = f"{dataset}-seed{i+1}"
        
        print(DATASET_NAME)
        try:
            train_gt = np.load(f"/home/deepstation/grasp-fractal/gqcnn-sim/3rdparty/dexnet/deps/gqcnn/analysis_real_6/{DATASET_NAME}/train_result.cres/labels.npz")["arr_0"]
            val_gt = np.load(f"/home/deepstation/grasp-fractal/gqcnn-sim/3rdparty/dexnet/deps/gqcnn/analysis_real_6/{DATASET_NAME}/val_result.cres/labels.npz")["arr_0"]
            train_pred = np.load(f"/home/deepstation/grasp-fractal/gqcnn-sim/3rdparty/dexnet/deps/gqcnn/analysis_real_6/{DATASET_NAME}/train_result.cres/predictions.npz")["arr_0"]
            val_pred = np.load(f"/home/deepstation/grasp-fractal/gqcnn-sim/3rdparty/dexnet/deps/gqcnn/analysis_real_6/{DATASET_NAME}/val_result.cres/predictions.npz")["arr_0"]

            train_indices = np.load("/home/deepstation/Downloads/realtrainset_6/splits/image_wise/train_indices.npz")["arr_0"]
            val_indices = np.load("/home/deepstation/Downloads/realtrainset_6/splits/image_wise/val_indices.npz")["arr_0"]
            indices = np.concatenate([train_indices, val_indices], axis=0)

    

            obj_labels_list = sorted(glob.glob("/home/deepstation/Downloads/realtrainset_6/tensors/obj_labels_*.npz"))
            obj_labels_all = None
            for obj_labels_file in obj_labels_list:
                obj_labels = np.load(obj_labels_file)["arr_0"]
                if obj_labels_all is None:
                    obj_labels_all = copy.copy(obj_labels)
                else:
                    obj_labels_all = np.concatenate([obj_labels_all, obj_labels], axis=0)
            obj_labels_pred = obj_labels_all[indices]
            
            

            obj_names = json.load(open("/home/deepstation/Downloads/realtrainset_6/object_category_map.json","rb"))
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
            pred_labels_obj = defaultdict(list)
            true_labels_obj = defaultdict(list)

            for t, p, y in zip(obj_labels_pred, pred_labels, true_labels):
                total_counts[t] += 1
                pred_labels_obj[t].append(p)
                true_labels_obj[t].append(y)

                if p == y:
                    correct_counts[t] += 1


            # 成功率を計算
            success_rate = {}
            for t in total_counts:
                success_rate[obj_ids[t]] = correct_counts[t] / total_counts[t]
                obj_list.append(obj_ids[t])
                success_rate_list.append(correct_counts[t] / total_counts[t])
                pricision_list.append(precision_score(true_labels_obj[t], pred_labels_obj[t]))

            #print(success_rate)
            #print(obj_list)
            #print(success_rate_list)
            #print(pricision_list)
            acc = accuracy_score(true_labels, pred_labels)
            prec = precision_score(true_labels, pred_labels)
            m_acc = sum(success_rate_list)/len(success_rate_list)
            m_prec = sum(pricision_list)/len(pricision_list)
            f1 = f1_score(true_labels, pred_labels)
            ap = average_precision_score(true_labels, pred_labels)
            # print(f"Accuracy:{acc}")
            # print(f"Precision:{prec}")
            # print(f"Macro-Accuracy:{m_acc}")
            # print(f"Macro-Precision:{m_prec}")
            # print(f"F1:{f1}")
            # print(f"AP:{ap}")
            overall_accuracy_list.append(acc)
            overall_pricision_list.append(prec)
            overall_macro_accuracy_list.append(m_acc)
            overall_macro_pricision_list.append(m_prec)
            overall_f1_list.append(f1)
            overall_ap_list.append(ap)


            #print(f"Accuracy: {accuracy:.4f} ({correct}/{total})")
            res.append(accuracy_score(true_labels, pred_labels))
        except Exception as e:
            print(e)
            continue
    #print(dataset)
    #print(res)
    print(f"Accuracy:{sum(overall_accuracy_list)/len(overall_accuracy_list)}")
    print(f"Precision:{sum(overall_pricision_list)/len(overall_pricision_list)}")
    print(f"Macro-Accuracy:{sum(overall_macro_accuracy_list)/len(overall_macro_accuracy_list)}")
    print(f"Macro-Precision:{sum(overall_macro_pricision_list)/len(overall_macro_pricision_list)}")
    print(f"F1:{sum(overall_f1_list)/len(overall_f1_list)}")
    print(f"AP:{sum(overall_ap_list)/len(overall_ap_list)}")


