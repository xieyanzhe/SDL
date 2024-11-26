import empyrical as ep
import numpy as np
import torch


def cal_return_ratios(preds, truths, last_prices):
    pred_ratios = []
    truth_ratios = []
    for i in range(preds.shape[0]):
        pred = preds[i].view(-1, 1)
        truth = truths[i].view(-1, 1)
        last_price = last_prices[i].view(-1, 1)

        return_ratio = torch.div(torch.sub(pred, last_price), last_price)
        truth_ratio = torch.div(torch.sub(truth, last_price), last_price)
        pred_ratios.append(return_ratio)
        truth_ratios.append(truth_ratio)

    pred_ratios = torch.stack(pred_ratios, dim=0).squeeze(-1)
    truth_ratios = torch.stack(truth_ratios, dim=0).squeeze(-1)
    return pred_ratios, truth_ratios


def cal_sharpe_ratio(pred_ratios, true_ratios, masks):
    return_ratios = []
    for i in range(pred_ratios.shape[1]):
        pred_ratio = pred_ratios[:, i]
        true_ratio = true_ratios[:, i]
        mask = masks[:, i].astype(int) > 0
        pred_ratio = pred_ratio[mask]
        true_ratio = true_ratio[mask]

        # 寻找pred_ratio最高的前5个在true_ratio中的位置
        rank_pred = np.argsort(-pred_ratio)
        selected_ratio = true_ratio[rank_pred[:5]]
        # rank_true = np.argsort(-true_ratio)
        # selected_ratio = true_ratio[rank_true[:5]]
        cur_return = np.sum(selected_ratio)
        return_ratios.append(cur_return)

    sharpe = ep.sharpe_ratio(np.array(return_ratios), period="daily")
    return sharpe


def cal_precision(pred_ratios, true_ratios, masks):
    precision = []
    for i in range(pred_ratios.shape[1]):
        pred_ratio = pred_ratios[:, i]
        true_ratio = true_ratios[:, i]
        mask = masks[:, i].astype(int) > 0
        pred_ratio = pred_ratio[mask]
        true_ratio = true_ratio[mask]

        # 寻找pred_ratio最高的前10个在true_ratio中的位置
        rank_pred = np.argsort(-pred_ratio)[:10]
        cur_precision = np.sum(true_ratio[rank_pred] >= 0) / 10
        precision.append(cur_precision)

    precision = np.mean(precision)
    return precision


def cal_irr(pred_ratios, true_ratios, masks):
    irr = []
    for i in range(pred_ratios.shape[1]):
        pred_ratio = pred_ratios[:, i]
        true_ratio = true_ratios[:, i]
        mask = masks[:, i].astype(int) > 0
        pred_ratio = pred_ratio[mask]
        true_ratio = true_ratio[mask]

        # 寻找pred_ratio最高的前5个在true_ratio中的位置
        rank_pred = np.argsort(-pred_ratio)
        selected_ratio = true_ratio[rank_pred[:5]]
        cur_return = np.sum(selected_ratio)
        irr.append(cur_return)

    irr = np.sum(irr)
    return irr


def cal_mrr(pred_ratios, true_ratios, masks):
    mrr = []
    for i in range(pred_ratios.shape[1]):
        pred_ratio = pred_ratios[:, i]
        true_ratio = true_ratios[:, i]
        mask = masks[:, i].astype(int) > 0
        pred_ratio = pred_ratio[mask]
        true_ratio = true_ratio[mask]

        # 寻找pred_ratio最高的前1个在true_ratio中的位置
        rank_pred = np.argsort(-pred_ratio)
        rank_true = np.argsort(-true_ratio)
        rank_pred_top1 = rank_pred[0]
        cur_mrr = 1 / (np.where(rank_true == rank_pred_top1)[0][0] + 1)
        mrr.append(cur_mrr)

    mrr = np.mean(mrr)
    return mrr


def cal_downside_div(pred_ratios, true_ratios, masks):
    downside_div = []
    for i in range(pred_ratios.shape[1]):
        pred_ratio = pred_ratios[:, i]
        true_ratio = true_ratios[:, i]
        mask = masks[:, i].astype(int) > 0
        pred_ratio = pred_ratio[mask]
        true_ratio = true_ratio[mask]

        # 寻找pred_ratio最高的前5个在true_ratio中的位置
        rank_pred = np.argsort(-pred_ratio)
        selected_ratio = true_ratio[rank_pred[:5]]
        cur_d = np.maximum(-selected_ratio, 0)
        cur_dd = np.sqrt(np.mean(np.square(cur_d)))
        downside_div.append(cur_dd)
    downside_div = np.mean(downside_div) * 15.87
    return downside_div


def cal_ic(pred_ratios, true_ratios, masks):
    ic = []
    for i in range(pred_ratios.shape[1]):
        pred_ratio = pred_ratios[:, i]
        true_ratio = true_ratios[:, i]
        mask = masks[:, i].astype(int)
        pred_ratio = pred_ratio[mask]
        true_ratio = true_ratio[mask]

        cur_ic = np.corrcoef(pred_ratio, true_ratio)[0, 1]
        ic.append(cur_ic)
    ic = np.mean(ic)
    return ic


def evaluate(pred_ratios, true_ratios, masks):
    metrics = ["sharpe@5", "irr@5", "precision@10", 'dd@5']  # "mrr"
    performance = {}

    for metric in metrics:
        if metric == 'sharpe@5':
            performance[metric] = cal_sharpe_ratio(pred_ratios, true_ratios, masks)
        elif metric == 'irr@5':
            performance[metric] = cal_irr(pred_ratios, true_ratios, masks)
        elif metric == 'precision@10':
            performance[metric] = cal_precision(pred_ratios, true_ratios, masks)
        elif metric == 'mrr':
            performance[metric] = cal_mrr(pred_ratios, true_ratios, masks)
        elif metric == 'dd@5':
            performance[metric] = cal_downside_div(pred_ratios, true_ratios, masks)
        elif metric == 'ic':
            performance[metric] = cal_ic(pred_ratios, true_ratios, masks)

    performance['total'] = 0
    for key, value in performance.items():
        if key != 'total' and key != 'dd@5':
            performance['total'] += value / (value / performance['sharpe@5'])
        elif key == 'dd@5':
            performance['total'] -= value / (value / performance['sharpe@5'])

    for key, value in performance.items():
        print(f"{key}: {value:.4f}", end=" | ")
    print()

    return performance
