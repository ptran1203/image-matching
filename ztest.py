def match_weight(f, model_type, fold, stage, loss_type):
    m, rest = f.split("_fold")
    match_model_type = model_type in {'auto', m}
    parts = rest.split("_")
    print(rest)
    _fold = int(parts[0].replace('fold', ''))
    _stage = int(parts[1].replace('stage', ''))
    _loss_type = parts[2]

    return match_model_type and _fold == fold and _stage == stage and _loss_type == loss_type


print(
    match_weight(
        'tf_efficientnet_b1_ns_fold3_stage2_cos_outdim779.pht',
        'auto', 3, 2, 'cos'
    )
)