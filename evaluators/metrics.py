import torch
from opacus.grad_sample.grad_sample_module import GradSampleModule

from utils import split_by_group

import numpy as np
from sklearn import metrics


def accuracy(model, dataloader, **kwargs):
    correct = 0
    total = 0
    with torch.no_grad():
        device = model._module.device if isinstance(model, GradSampleModule) else model.device
        for _batch_idx, (data, labels, group) in enumerate(dataloader):
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()
    return (correct / total).item()


def accuracy_per_group(model, dataloader, num_groups=None, **kwargs):
    correct_per_group = [0] * num_groups
    total_per_group = [0] * num_groups
    with torch.no_grad():
        device = model._module.device if isinstance(model, GradSampleModule) else model.device
        for _batch_idx, (data, labels, group) in enumerate(dataloader):
            data, labels = data.to(device), labels.to(device)

            per_group = split_by_group(data, labels, group, num_groups)
            for i, group in enumerate(per_group):
                data_group, labels_group = group
                outputs = model(data_group)
                _, predicted = torch.max(outputs, 1)
                total_per_group[i] += labels_group.size(0)
                correct_per_group[i] += (predicted == labels_group).sum()
    # print([float(correct_per_group[i] / total_per_group[i]) for i in range(num_groups)])
    return [float(correct_per_group[i] / total_per_group[i]) for i in range(num_groups)]


def macro_accuracy(model, dataloader, num_classes=None, **kwargs):
    confusion_matrix = torch.zeros(num_classes, num_classes)
    with torch.no_grad():
        device = model._module.device if isinstance(model, GradSampleModule) else model.device
        for _batch_idx, (data, labels, group) in enumerate(dataloader):
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            for true_p, all_p in zip(labels.view(-1), predicted.view(-1)):
                confusion_matrix[true_p.long(), all_p.long()] += 1
    accs = confusion_matrix.diag() / confusion_matrix.sum(1)
    # print(accs)
    return accs.mean().item()

def roc_auc(model, dataloader, **kwargs):
    with torch.no_grad():
        device = model._module.device if isinstance(model, GradSampleModule) else model.device
        correct_labels = []
        predict_probs = []
        for _batch_idx, (data, labels, group) in enumerate(dataloader):
            correct_labels.extend([x.item() for x in labels])
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            if kwargs["num_classes"] > 2: 
                predict_probs.extend(torch.softmax(outputs,1).cpu())
            else:
                predict_probs.extend([x.item() for x in torch.softmax(outputs,1)[:,1].cpu()])
    if kwargs["num_classes"] > 2: 
        return metrics.roc_auc_score(correct_labels, predict_probs, multi_class='ovr')
    else:
        return metrics.roc_auc_score(correct_labels, predict_probs)

def roc_auc_per_group(model, dataloader, num_groups=None, **kwargs):
    with torch.no_grad():
        device = model._module.device if isinstance(model, GradSampleModule) else model.device
        correct_labels = []
        predict_probs = []
        groups = []
        for _batch_idx, (data, labels, group) in enumerate(dataloader):
            correct_labels.extend([x.item() for x in labels])
            groups.extend([x.item() for x in group])
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            if kwargs["num_classes"] > 2: 
                predict_probs.extend(torch.softmax(outputs,1).cpu())
            else:
                predict_probs.extend([x.item() for x in torch.softmax(outputs,1).cpu()[:,1]])
    if kwargs["num_classes"] == 10 and num_groups==10: #i.e., if MNIST
        return metrics.roc_auc_score(np.array(correct_labels), np.array(predict_probs), multi_class='ovr', average=None)
    else:
        roc_auc_group0 = metrics.roc_auc_score(np.array(correct_labels)[~np.array(groups).astype(bool)], np.array(predict_probs)[~np.array(groups).astype(bool)])
        roc_auc_group1 = metrics.roc_auc_score(np.array(correct_labels)[np.array(groups).astype(bool)], np.array(predict_probs)[np.array(groups).astype(bool)])
        return [roc_auc_group0, roc_auc_group1]

def pr_auc(model, dataloader, **kwargs):
    with torch.no_grad():
        device = model._module.device if isinstance(model, GradSampleModule) else model.device
        correct_labels = []
        predict_probs = []
        for _batch_idx, (data, labels, group) in enumerate(dataloader):
            correct_labels.extend([x.item() for x in labels])
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            if kwargs["num_classes"] > 2: 
                predict_probs.extend(torch.softmax(outputs,1).cpu())
            else:
                predict_probs.extend([x.item() for x in torch.softmax(outputs,1).cpu()[:,1]])
    if kwargs["num_classes"] > 2:
        pr_aucs = [0]*kwargs["num_classes"]
        for i in range(kwargs["num_classes"]):
            precisions, recalls, thresholds = metrics.precision_recall_curve((np.array(correct_labels)==i).astype(int), np.array(predict_probs)[:,i])
            pr_aucs[i] = metrics.auc(recalls, precisions)
        return np.mean(pr_aucs)
    else:
        precisions, recalls, thresholds = metrics.precision_recall_curve(correct_labels, predict_probs)
        return metrics.auc(recalls, precisions)

def pr_auc_per_group(model, dataloader, num_groups=None, **kwargs):
    with torch.no_grad():
        device = model._module.device if isinstance(model, GradSampleModule) else model.device
        correct_labels = []
        predict_probs = []
        groups = []
        for _batch_idx, (data, labels, group) in enumerate(dataloader):
            correct_labels.extend([x.item() for x in labels])
            groups.extend([x.item() for x in group])
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            if kwargs["num_classes"] > 2: 
                predict_probs.extend(torch.softmax(outputs,1).cpu())
            else:
                predict_probs.extend([x.item() for x in torch.softmax(outputs,1).cpu()[:,1]])
    if num_groups == 10 and kwargs["num_classes"]==10:
        pr_aucs = [0]*kwargs["num_classes"]
        for i in range(kwargs["num_classes"]):
            precisions, recalls, thresholds = metrics.precision_recall_curve((np.array(correct_labels)==i).astype(int), np.array(predict_probs)[:,i])
            pr_aucs[i] = metrics.auc(recalls, precisions)
        return pr_aucs
    else:
        precisions0, recalls0, thresholds0 = metrics.precision_recall_curve(np.array(correct_labels)[~np.array(groups).astype(bool)], np.array(predict_probs)[~np.array(groups).astype(bool)])
        pr_auc0 = metrics.auc(recalls0, precisions0)
        precisions1, recalls1, thresholds1 = metrics.precision_recall_curve(np.array(correct_labels)[np.array(groups).astype(bool)], np.array(predict_probs)[np.array(groups).astype(bool)])
        pr_auc1 = metrics.auc(recalls1, precisions1)
        return [pr_auc0, pr_auc1]

def recall(model, dataloader, **kwargs):
    with torch.no_grad():
        device = model._module.device if isinstance(model, GradSampleModule) else model.device
        correct_labels = []
        predict_labels = []
        for _batch_idx, (data, labels, group) in enumerate(dataloader):
            correct_labels.extend([x.item() for x in labels])
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            predict_labels.extend([x.item() for x in predicted])
    if kwargs["num_classes"]==10:
        return metrics.recall_score(correct_labels, predict_labels, average='macro')
    else:
        tn, fp, fn, tp = metrics.confusion_matrix(correct_labels, predict_labels).ravel()
        return tp/(tp+fn)

def recall_per_group(model, dataloader, num_groups=None, **kwargs):
    with torch.no_grad():
        device = model._module.device if isinstance(model, GradSampleModule) else model.device
        correct_labels = []
        predict_labels = []
        groups = []
        for _batch_idx, (data, labels, group) in enumerate(dataloader):
            correct_labels.extend([x.item() for x in labels])
            groups.extend([x.item() for x in group])
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            predict_labels.extend([x.item() for x in predicted])
    if num_groups == 10 and kwargs["num_classes"]==10:
        # recalls = [0]*num_groups
        # for i in range(num_groups):
        #     tn, fp, fn, tp = metrics.confusion_matrix((np.array(correct_labels) == i).astype(int), (np.array(predict_labels)==i).astype(int)).ravel()
        #     recalls[i] = tp/(tp+fn)
        return metrics.recall_score(correct_labels, predict_labels, average=None)
    else:
        tn0, fp0, fn0, tp0 = metrics.confusion_matrix(np.array(correct_labels)[~np.array(groups).astype(bool)], np.array(predict_labels)[~np.array(groups).astype(bool)]).ravel()
        recall0 = tp0/(tp0+fn0)
        tn1, fp1, fn1, tp1 = metrics.confusion_matrix(np.array(correct_labels)[np.array(groups).astype(bool)], np.array(predict_labels)[np.array(groups).astype(bool)]).ravel()
        recall1 = tp1/(tp1+fn1)
        return [recall0, recall1]

def precision(model, dataloader, **kwargs):
    with torch.no_grad():
        device = model._module.device if isinstance(model, GradSampleModule) else model.device
        correct_labels = []
        predict_labels = []
        for _batch_idx, (data, labels, group) in enumerate(dataloader):
            correct_labels.extend([x.item() for x in labels])
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            predict_labels.extend([x.item() for x in predicted])
    if kwargs["num_classes"]==10:
        return metrics.precision_score(correct_labels, predict_labels, average='macro')
    else:
        tn, fp, fn, tp = metrics.confusion_matrix(correct_labels, predict_labels).ravel()
        return tp/(tp+fp)

def acceptance_rate_per_group(model, dataloader, num_groups=None, **kwargs):
    with torch.no_grad():
        device = model._module.device if isinstance(model, GradSampleModule) else model.device
        correct_labels = []
        predict_labels = []
        groups = []
        for _batch_idx, (data, labels, group) in enumerate(dataloader):
            correct_labels.extend([x.item() for x in labels])
            groups.extend([x.item() for x in group])
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            predict_labels.extend([x.item() for x in predicted])
    if num_groups == 10 and kwargs["num_classes"]==10:
        acceptance_rates = [0]*num_groups
        for i in range(num_groups):
            tn, fp, fn, tp = metrics.confusion_matrix((np.array(correct_labels) == i).astype(int), (np.array(predict_labels)==i).astype(int)).ravel()
            acceptance_rates[i] = (tp+fp)/(np.array(correct_labels) == i).sum()
        return acceptance_rates
    else:
        tn0, fp0, fn0, tp0 = metrics.confusion_matrix(np.array(correct_labels)[~np.array(groups).astype(bool)], np.array(predict_labels)[~np.array(groups).astype(bool)]).ravel()
        tn1, fp1, fn1, tp1 = metrics.confusion_matrix(np.array(correct_labels)[np.array(groups).astype(bool)], np.array(predict_labels)[np.array(groups).astype(bool)]).ravel()
        acceptance_rate0 = (tp0+fp0)/(len(groups)-np.sum(groups))
        acceptance_rate1 = (tp1+fp1)/np.sum(groups)
        return [acceptance_rate0, acceptance_rate1]

def treatment_per_group(model, dataloader, num_groups=None, **kwargs):
    with torch.no_grad():
        device = model._module.device if isinstance(model, GradSampleModule) else model.device
        correct_labels = []
        predict_labels = []
        groups = []
        for _batch_idx, (data, labels, group) in enumerate(dataloader):
            correct_labels.extend([x.item() for x in labels])
            groups.extend([x.item() for x in group])
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            predict_labels.extend([x.item() for x in predicted])
    if num_groups == 10 and kwargs["num_classes"]==10:
        treatments = [0]*num_groups
        for i in range(num_groups):
            tn, fp, fn, tp = metrics.confusion_matrix((np.array(correct_labels) == i).astype(int), (np.array(predict_labels)==i).astype(int)).ravel()
            treatments[i] = fn/fp
        return treatments
    else:
        tn0, fp0, fn0, tp0 = metrics.confusion_matrix(np.array(correct_labels)[~np.array(groups).astype(bool)], np.array(predict_labels)[~np.array(groups).astype(bool)]).ravel()
        tn1, fp1, fn1, tp1 = metrics.confusion_matrix(np.array(correct_labels)[np.array(groups).astype(bool)], np.array(predict_labels)[np.array(groups).astype(bool)]).ravel()
        treatment0 = fn0/fp0
        treatment1 = fn1/fp1
        return [treatment0, treatment1]

def PPV_per_group(model, dataloader, num_groups=None, **kwargs):
    with torch.no_grad():
        device = model._module.device if isinstance(model, GradSampleModule) else model.device
        correct_labels = []
        predict_labels = []
        groups = []
        for _batch_idx, (data, labels, group) in enumerate(dataloader):
            correct_labels.extend([x.item() for x in labels])
            groups.extend([x.item() for x in group])
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            predict_labels.extend([x.item() for x in predicted])
    if num_groups == 10 and kwargs["num_classes"]==10:
        # PPVs = [0]*num_groups
        # for i in range(num_groups):
        #     tn, fp, fn, tp = metrics.confusion_matrix((np.array(correct_labels) == i).astype(int), (np.array(predict_labels)==i).astype(int)).ravel()
        #     PPVs[i] = tp/(tp+fp)
        return metrics.precision_score(correct_labels, predict_labels, average=None)
    else:
        tn0, fp0, fn0, tp0 = metrics.confusion_matrix(np.array(correct_labels)[~np.array(groups).astype(bool)], np.array(predict_labels)[~np.array(groups).astype(bool)]).ravel()
        tn1, fp1, fn1, tp1 = metrics.confusion_matrix(np.array(correct_labels)[np.array(groups).astype(bool)], np.array(predict_labels)[np.array(groups).astype(bool)]).ravel()
        PPV0 = tp0/(tp0+fp0)
        PPV1 = tp1/(tp1+fp1)
        return [PPV0, PPV1]

def FP_errorrate_per_group(model, dataloader, num_groups=None, **kwargs):
    with torch.no_grad():
        device = model._module.device if isinstance(model, GradSampleModule) else model.device
        correct_labels = []
        predict_labels = []
        groups = []
        for _batch_idx, (data, labels, group) in enumerate(dataloader):
            correct_labels.extend([x.item() for x in labels])
            groups.extend([x.item() for x in group])
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            predict_labels.extend([x.item() for x in predicted])
    if num_groups == 10 and kwargs["num_classes"]==10:
        FP_errorrates = [0]*num_groups
        for i in range(num_groups):
            tn, fp, fn, tp = metrics.confusion_matrix((np.array(correct_labels) == i).astype(int), (np.array(predict_labels)==i).astype(int)).ravel()
            FP_errorrates[i] = fp/(fp+tn)
        return FP_errorrates
    else:
        tn0, fp0, fn0, tp0 = metrics.confusion_matrix(np.array(correct_labels)[~np.array(groups).astype(bool)], np.array(predict_labels)[~np.array(groups).astype(bool)]).ravel()
        tn1, fp1, fn1, tp1 = metrics.confusion_matrix(np.array(correct_labels)[np.array(groups).astype(bool)], np.array(predict_labels)[np.array(groups).astype(bool)]).ravel()
        FP_errorrate0 = fp0/(fp0+tn0)
        FP_errorrate1 = fp1/(fp1+tn1)
        return [FP_errorrate0, FP_errorrate1]

def FN_errorrate_per_group(model, dataloader, num_groups=None, **kwargs):
    with torch.no_grad():
        device = model._module.device if isinstance(model, GradSampleModule) else model.device
        correct_labels = []
        predict_labels = []
        groups = []
        for _batch_idx, (data, labels, group) in enumerate(dataloader):
            correct_labels.extend([x.item() for x in labels])
            groups.extend([x.item() for x in group])
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            predict_labels.extend([x.item() for x in predicted])
    if num_groups == 10 and kwargs["num_classes"]==10:
        FN_errorrates = [0]*num_groups
        for i in range(num_groups):
            tn, fp, fn, tp = metrics.confusion_matrix((np.array(correct_labels) == i).astype(int), (np.array(predict_labels)==i).astype(int)).ravel()
            FN_errorrates[i] = fn/(tp+fn)
        return FN_errorrates
    else:
        tn0, fp0, fn0, tp0 = metrics.confusion_matrix(np.array(correct_labels)[~np.array(groups).astype(bool)], np.array(predict_labels)[~np.array(groups).astype(bool)]).ravel()
        tn1, fp1, fn1, tp1 = metrics.confusion_matrix(np.array(correct_labels)[np.array(groups).astype(bool)], np.array(predict_labels)[np.array(groups).astype(bool)]).ravel()
        FN_errorrate0 = fn0/(tp0+fn0)
        FN_errorrate1 = fn1/(tp1+fn1)
        return [FN_errorrate0, FN_errorrate1]