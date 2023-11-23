import torch


def compute_super_CP_multilabel_loss(pred_logits, CP_mega_matrices):
    logits = []
    labels = []
    bs, n_relations, _, _ = pred_logits.shape
    for i in range(bs):
        pred_logit = pred_logits[i, :, :, :].permute(
            0, 2, 1
        )  # n_relations, N, n_mega_voxels
        CP_mega_matrix = CP_mega_matrices[i]  # n_relations, N, n_mega_voxels
        logits.append(pred_logit.reshape(n_relations, -1))
        labels.append(CP_mega_matrix.reshape(n_relations, -1))

    logits = torch.cat(logits, dim=1).T  # M, 4
    labels = torch.cat(labels, dim=1).T  # M, 4

    cnt_neg = (labels == 0).sum(0)
    cnt_pos = labels.sum(0)
    pos_weight = cnt_neg / cnt_pos
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    loss_bce = criterion(logits, labels.float())
    return loss_bce
