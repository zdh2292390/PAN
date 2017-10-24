# -*- coding: utf-8 -*-
from evaluate import strict, loose_macro, loose_micro

def get_true_and_prediction(scores, y_data, gaussian=False, delta=0, topK=1, use_path=False, label_hierarchy=None, alpha=0.5):
    true_and_prediction = []
    if gaussian == False:
        for score,true_label in zip(scores,y_data):
            predicted_tag = []
            true_tag = []
            for label_id,label_score in enumerate(list(true_label)):
                if label_score > 0:
                    true_tag.append(label_id)
                    if use_path:
                        label_id_tmp = label_id
                        while len(label_hierarchy[label_id_tmp]['father'])!=0:
                            label_id_tmp = label_hierarchy[label_id_tmp]['father'][0]
                            true_tag.append(label_id_tmp)
            true_tag = list(set(true_tag))

            lid,ls = max(enumerate(list(score)),key=lambda x: x[1])
            predicted_tag.append(lid)
            for label_id,label_score in enumerate(list(score)):
                if label_score > alpha:
                    if label_id != lid:
                        predicted_tag.append(label_id)
                        if use_path:
                            label_id_tmp = label_id
                            while len(label_hierarchy[label_id_tmp]['father'])!=0:
                                label_id_tmp = label_hierarchy[label_id_tmp]['father'][0]
                                predicted_tag.append(label_id_tmp)

            predicted_tag = list(set(predicted_tag))
            true_and_prediction.append((true_tag, predicted_tag))

    else:
        print "K:"+str(topK)
        for score,true_label in zip(scores,y_data):
            predicted_tag = []
            true_tag = []
            for label_id,label_score in enumerate(list(true_label)):
                if label_score > 0:
                    true_tag.append(label_id)
            sorted_score = sorted(enumerate(list(score)),key=lambda x: x[1])
            level1 = score[0:4]
            sorted_level1 = sorted(enumerate(list(level1)),key=lambda x: x[1])
            for x in range(topK):
                predicted_tag.append(sorted_level1[x][0])
            # print "test:"+str(len(predicted_tag))
            for label_id,label_score in enumerate(list(score)):
                if label_score < delta:
                    if label_id not in predicted_tag:
                        predicted_tag.append(label_id)
            true_and_prediction.append((true_tag, predicted_tag))
            
    return true_and_prediction

def acc_hook(scores, y_data, gaussian=False, delta=0, topK=1, use_path=False, label_hierarchy=None):
    true_and_prediction = get_true_and_prediction(scores, y_data, gaussian, delta, topK, use_path,label_hierarchy)
    print("     strict (p,r,f1):",strict(true_and_prediction))
    print("loose macro (p,r,f1):",loose_macro(true_and_prediction))
    print("loose micro (p,r,f1):",loose_micro(true_and_prediction))


def save_predictions(scores, y_data, id2label, fname, gaussian=False):
    true_and_prediction = get_true_and_prediction(scores, y_data, gaussian)
    with open(fname,"w") as f:
        for t, p in true_and_prediction:
            f.write(" ".join([id2label[id] for id in t]) + "\t" + " ".join([id2label[id] for id in p]) + "\n")
    f.close()

# def statistics():