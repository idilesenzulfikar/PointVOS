#!/usr/bin/env python3

"""
TODO: There are two keys files one of them in dataset_scripts and the other is in the method. Consider merging them in the future.
"""

from enum import Enum


# ===================================
# Enums for options
# ===================================
class AnnotationSupervision(Enum):
    one_point = "1-point"
    two_point = "2-point"
    three_point = "3-point"
    four_point = "4-point"
    five_point = "5-point"
    ten_point = "10-point"
    twenty_point = "20-point"
    thirty_point = "30-point"
    mask = "mask"

class Dataset(Enum):
    oops = "Oops"
    kinetics = "Kinetics"
    youtube = "YouTube"
    davis = "DAVIS"

# ============================================
# Convert string to Enum
# ============================================
def str_to_annotation_supervision(ann_sup: str):
    if ann_sup == AnnotationSupervision.one_point.value:
        return AnnotationSupervision.one_point
    elif ann_sup == AnnotationSupervision.two_point.value:
        return AnnotationSupervision.two_point
    elif ann_sup == AnnotationSupervision.three_point.value:
        return AnnotationSupervision.three_point
    elif ann_sup == AnnotationSupervision.four_point.value:
        return AnnotationSupervision.four_point
    elif ann_sup == AnnotationSupervision.five_point.value:
        return AnnotationSupervision.five_point
    elif ann_sup == AnnotationSupervision.ten_point.value:
        return AnnotationSupervision.ten_point
    elif ann_sup == AnnotationSupervision.twenty_point.value:
        return AnnotationSupervision.twenty_point
    elif ann_sup == AnnotationSupervision.thirty_point.value:
        return AnnotationSupervision.thirty_point
    elif ann_sup == AnnotationSupervision.random_point.value:
        return AnnotationSupervision.random_point
    elif ann_sup == AnnotationSupervision.mask.value:
        return AnnotationSupervision.mask
    else:
        raise ValueError(
            "Invalid annotation supervision {0}. The valid annotation supervisions are {1}.".format(ann_sup,
                                                                                                    [e.value for e in
                                                                                                     AnnotationSupervision]))

def str_to_dataset(dataset: str):
    if dataset == Dataset.oops.value:
        return Dataset.oops
    elif dataset == Dataset.kinetics.value:
        return Dataset.kinetics
    elif dataset == Dataset.youtube.value:
        return Dataset.youtube
    elif dataset == Dataset.davis.value:
        return Dataset.davis
    else:
        raise ValueError(
            "Invalid dataset name {0}. The valid dataset names are {1}".format(dataset, [e.value for e in Dataset]))

# =================================================
# Convert Enum to string
# =================================================
def annotation_supervision_to_str(ann_sup: AnnotationSupervision):
    if ann_sup == AnnotationSupervision.one_point:
        return AnnotationSupervision.one_point.value
    elif ann_sup == AnnotationSupervision.two_point:
        return AnnotationSupervision.two_point.value
    elif ann_sup == AnnotationSupervision.three_point:
        return AnnotationSupervision.three_point.value
    elif ann_sup == AnnotationSupervision.four_point:
        return AnnotationSupervision.four_point.value
    elif ann_sup == AnnotationSupervision.five_point:
        return AnnotationSupervision.five_point.value
    elif ann_sup == AnnotationSupervision.ten_point:
        return AnnotationSupervision.ten_point.value
    elif ann_sup == AnnotationSupervision.twenty_point:
        return AnnotationSupervision.twenty_point.value
    elif ann_sup == AnnotationSupervision.thirty_point:
        return AnnotationSupervision.thirty_point.value
    elif ann_sup == AnnotationSupervision.random_point.value:
        return AnnotationSupervision.random_point.value
    elif ann_sup == AnnotationSupervision.mask:
        return AnnotationSupervision.mask.value
    else:
        raise TypeError(
            "Invalid Annotation Supervision object {0}. The valid Annotation Supervision objects are {1}.".format(
                ann_sup, [e for e in AnnotationSupervision]))

def dataset_to_str(dataset: Dataset):
    if dataset == Dataset.oops:
        return Dataset.oops.value
    elif dataset == Dataset.kinetics:
        return Dataset.kinetics.value
    elif dataset == Dataset.youtube:
        return Dataset.youtube.value
    elif dataset == Dataset.davis:
        return Dataset.davis.value
    else:
        raise TypeError(
            "Invalid Dataset object {0}. The valid Dataset objects are {1}".format(dataset, [e for e in Dataset]))
