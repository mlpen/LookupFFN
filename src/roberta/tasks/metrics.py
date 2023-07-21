import torch.nn as nn
import torch
import math

class Accuracy(nn.Module):
    def __init__(self, ignore_index = -100):
        super().__init__()
        self.ignore_index = ignore_index

    def __call__(self, scores, labels):
        mask = (labels != self.ignore_index).float()
        correct = scores.argmax(dim = -1) == labels
        correct = (correct.float() * mask).sum()
        count = mask.sum()
        accu = correct / (count + 1e-6)
        return accu.to(scores.dtype), count.to(scores.dtype)

class Loss(nn.Module):
    def __init__(self, ignore_index = -100):
        super().__init__()
        self.ignore_index = ignore_index
        self.loss_fct = nn.CrossEntropyLoss(reduction = "none")

    def __call__(self, scores, labels):
        mask = (labels != self.ignore_index).float()
        loss = self.loss_fct(scores, labels).float()
        loss = (loss * mask).sum()
        count = mask.sum()
        loss = loss / (count + 1e-6)
        return loss.to(scores.dtype), count.to(scores.dtype)

class F1(nn.Module):
    def __init__(self, ignore_index = -100):
        super().__init__()
        self.ignore_index = ignore_index

    def __call__(self, scores, labels):
        mask = (labels != self.ignore_index).to(scores.dtype)
        count = mask.sum()
        pred = scores.argmax(dim = -1)
        tp = (labels * pred * mask).sum().to(scores.dtype)
        tn = ((1 - labels) * (1 - pred) * mask).sum().to(scores.dtype)
        fp = ((1 - labels) * pred * mask).sum().to(scores.dtype)
        fn = (labels * (1 - pred) * mask).sum().to(scores.dtype)
        epsilon = 1e-7
        precision = tp / (tp + fp + epsilon)
        recall = tp / (tp + fn + epsilon)

        f1 = 2 * (precision * recall) / (precision + recall + epsilon)
        f1 = f1.clamp(min = epsilon, max = 1 - epsilon)

        return f1, count
