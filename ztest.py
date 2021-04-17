import re


def get_info_from_weight(path):
    """
    Inputs form: {backbone}_fold{fold}_stage{stage}_{loss_type}_outdim{outdim}.pth
    e.g: tf_efficientnet_b1_ns_fold0_stage1_cos_outdim11014.pth
    """
    def reformat(value):
        try:
            return int(value)
        except:
            # if contains number, get number
            numbers = re.findall(r'[0-9]+', value)
            if numbers:
                return int(numbers[0])
            return value

    # Get backbone first
    backbone, parts = path.split('_fold')
    fold, stage, loss_type, outdim = [reformat(v) for v in parts.split("_")]

    return backbone, fold, stage, loss_type, outdim

for path in [
    'tf_efficientnet_b1_ns_fold0_stage3_cos_outdim11014.pth',
    'resnest50_fold2_stage2_arc_outdim8811.pth'
]:
    print(get_info_from_weight(path))