import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


NEGATIVE_LEVEL = 'ND'
RECALL_K = 80
RECALL_K_LIST = [10, 20, 50, 100]
MVEP_K_LIST = [20, 50, 80, 100]


def load_predictions(path='nodes_info_with_inference100.csv'):
    df = pd.read_csv(path)
    if 'Node' not in df.columns or 'bayes_normalized' not in df.columns:
        raise ValueError("nodes_info_with_inference100.csv must contain 'Node' and 'bayes_normalized'.")
    df = df[['Node', 'bayes_normalized']].copy()
    df['score'] = df['bayes_normalized'].astype(float) / 100.0
    return df[['Node', 'score']]


def load_edges_level(path='edges_level.csv'):
    df = pd.read_csv(path)
    required = {'Up', 'UpScreeningResult', 'Down', 'DownScreeningResult', 'Name'}
    if not required.issubset(df.columns):
        raise ValueError("edges_level.csv must contain Up, UpScreeningResult, Down, DownScreeningResult, Name.")
    return df


def load_edges_info(path='edges_info.csv'):
    df = pd.read_csv(path)
    required = {'Source', 'Target', 'Mechanism'}
    if not required.issubset(df.columns):
        raise ValueError("edges_info.csv must contain Source, Target, Mechanism.")
    return df[['Source', 'Target', 'Mechanism']]


def recall_at_k(sorted_flags, total_pos, k):
    if total_pos == 0:
        return np.nan
    k = min(k, len(sorted_flags))
    hits = int(np.sum(sorted_flags[:k]))
    return hits / total_pos


def mrr_from_ranks(ranks):
    if not ranks:
        return np.nan
    return float(np.mean([1.0 / r for r in ranks]))


def macro_f1(y_true, y_pred):
    labels = sorted(set(y_true) | set(y_pred))
    f1s = []
    for label in labels:
        tp = sum((yt == label) and (yp == label) for yt, yp in zip(y_true, y_pred))
        fp = sum((yt != label) and (yp == label) for yt, yp in zip(y_true, y_pred))
        fn = sum((yt == label) and (yp != label) for yt, yp in zip(y_true, y_pred))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
        f1s.append(f1)
    return float(np.mean(f1s)) if f1s else np.nan


def plot_down_rank_distribution(down_df, k, enrichment_at_k, out_path='bn_benchmark_down_rank.png'):
    down_df = down_df.sort_values('down_score', ascending=False).reset_index(drop=True)
    ranks = np.arange(1, len(down_df) + 1)
    colors = np.where(down_df['is_pos'], '#D55E00', '#9E9E9E')
    plt.figure(figsize=(8, 5))
    plt.scatter(ranks, down_df['down_score'], s=18, c=colors, alpha=0.9)
    if k is not None:
        plt.axvline(k, color='#666666', linestyle='--', linewidth=1)
    if enrichment_at_k is not None and not np.isnan(enrichment_at_k):
        plt.text(0.02, 0.95, f'Enrichment@{k} = {enrichment_at_k:.2f}x',
                 transform=plt.gca().transAxes, fontsize=10,
                 bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
    plt.xlabel('Rank (Down score, descending)')
    plt.ylabel('Pred node score')
    plt.title('Down Node Score Ranking')
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_recall_at_k_curve(flags, total_pos, ks, out_path='bn_benchmark_recall_at_k.png'):
    recalls = []
    for k in ks:
        recalls.append(recall_at_k(flags, total_pos, k))
    plt.figure(figsize=(7, 5))
    plt.plot(ks, recalls, marker='o', color='#1B9E77', linewidth=2)
    plt.xlabel('K (Top-K)')
    plt.ylabel('Recall@K')
    plt.title('Recall@K Curve (Observed Down)')
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_mechanism_confusion(cm_df, out_path='bn_benchmark_mechanism_cm.png', normalize=True):
    if cm_df.empty:
        return
    cm = cm_df.copy()
    if normalize:
        cm = cm.div(cm.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm.values, aspect='auto', cmap='Blues')
    plt.colorbar(label='Proportion' if normalize else 'Count')
    plt.xticks(ticks=np.arange(len(cm.columns)), labels=cm.columns, rotation=90)
    plt.yticks(ticks=np.arange(len(cm.index)), labels=cm.index)
    plt.xlabel('Pred Mechanism')
    plt.ylabel('True Mechanism')
    plt.title('Mechanism Confusion Matrix')
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_mvep_at_k(mvep_map, out_path='bn_benchmark_mvep_at_k.png'):
    if not mvep_map:
        return
    ks = sorted(mvep_map.keys())
    values = [mvep_map[k] for k in ks]
    plt.figure(figsize=(6.5, 4.5))
    plt.bar([str(k) for k in ks], values, color='#4C72B0')
    plt.ylim(0, 1)
    plt.xlabel('K (Top-K edges)')
    plt.ylabel('MVEP@K')
    plt.title('Mechanism-Verified Edge Precision@K')
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_two_stage_summary(summary, out_path='bn_benchmark_two_stage.png'):
    stage1_vals = [summary['ods_recall_k'], summary['ods_mrr'], summary['ods']]
    stage2_vals = [summary['bems_macro_f1_mech'], summary['bems_edge_recall_k'], summary['bems']]
    stage1_labels = ['Recall@K', 'MRR', 'ODS']
    stage2_labels = ['Macro-F1', 'EdgeRecall@K', 'BEMS']

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].bar(stage1_labels, stage1_vals, color='#1B9E77')
    axes[0].set_ylim(0, 1)
    axes[0].set_title('Stage 1 (Discovery)')
    axes[0].set_ylabel('Score')

    axes[1].bar(stage2_labels, stage2_vals, color='#D95F02')
    axes[1].set_ylim(0, 1)
    axes[1].set_title('Stage 2 (Mechanism)')

    axes[2].bar(['TSOS'], [summary['tsos']], color='#7570B3')
    axes[2].set_ylim(0, 1)
    axes[2].set_title('Overall')

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def main():
    preds = load_predictions()
    edges_level = load_edges_level()
    edges_info = load_edges_info()

    edges_level = edges_level.merge(
        edges_info,
        left_on=['Up', 'Down'],
        right_on=['Source', 'Target'],
        how='left'
    )

    preds_map = dict(zip(preds['Node'], preds['score']))
    edges_level['down_score'] = edges_level['Down'].map(preds_map)
    edges_level['up_score'] = edges_level['Up'].map(preds_map)
    edges_level['edge_score'] = edges_level[['up_score', 'down_score']].mean(axis=1)

    # Benchmark-1: Observed Discovery Score (ODS)
    down_df = edges_level[['Down', 'DownScreeningResult', 'down_score']].drop_duplicates('Down')
    down_df = down_df.dropna(subset=['down_score'])
    down_df['is_pos'] = down_df['DownScreeningResult'].apply(lambda x: str(x).strip() != NEGATIVE_LEVEL)

    if down_df.empty or down_df['is_pos'].sum() == 0:
        ods_recall_k = np.nan
        ods_mrr = np.nan
        enrichment_at_k = np.nan
        precision_at_k = np.nan
        flags = np.array([], dtype=int)
        total_pos = 0
    else:
        down_df = down_df.sort_values('down_score', ascending=False).reset_index(drop=True)
        flags = down_df['is_pos'].astype(int).to_numpy()
        total_pos = int(flags.sum())
        ods_recall_k = recall_at_k(flags, total_pos, RECALL_K)
        pos_ranks = [idx + 1 for idx, flag in enumerate(flags) if flag == 1]
        ods_mrr = mrr_from_ranks(pos_ranks)
        hits = int(np.sum(flags[:min(RECALL_K, len(flags))]))
        precision_at_k = hits / min(RECALL_K, len(flags))
        pos_rate = total_pos / max(len(flags), 1)
        enrichment_at_k = (precision_at_k / pos_rate) if pos_rate > 0 else np.nan

    ods = 0.5 * ods_recall_k + 0.5 * ods_mrr if not np.isnan(ods_recall_k) and not np.isnan(ods_mrr) else np.nan

    # Benchmark-2: Both-Observed Edge Mechanism Score (BEMS)
    gold_mask = (edges_level['UpScreeningResult'].apply(lambda x: str(x).strip() != NEGATIVE_LEVEL)) & (
        edges_level['DownScreeningResult'].apply(lambda x: str(x).strip() != NEGATIVE_LEVEL)
    )
    all_edges = edges_level.dropna(subset=['edge_score']).copy()
    all_edges['is_gold'] = gold_mask.loc[all_edges.index]
    gold_edges = all_edges[all_edges['is_gold']]

    if all_edges.empty or gold_edges.empty:
        edge_recall_k = np.nan
        macro_f1_mech = np.nan
        bems = np.nan
        mech_cm = pd.DataFrame()
        mvep_map = {}
    else:
        all_edges = all_edges.sort_values('edge_score', ascending=False).reset_index(drop=True)
        flags = all_edges['is_gold'].astype(int).to_numpy()
        total_gold = int(flags.sum())
        edge_recall_k = recall_at_k(flags, total_gold, RECALL_K)

        y_true_mech = gold_edges['Name'].fillna('UNKNOWN').astype(str).tolist()
        y_pred_mech = gold_edges['Mechanism'].fillna('UNKNOWN').astype(str).tolist()
        macro_f1_mech = macro_f1(y_true_mech, y_pred_mech)

        mech_cm = pd.crosstab(
            pd.Series(y_true_mech, name='true'),
            pd.Series(y_pred_mech, name='pred')
        )

        all_edges['mechanism_true'] = all_edges['Name'].fillna('UNKNOWN').astype(str)
        all_edges['mechanism_pred'] = all_edges['Mechanism'].fillna('UNKNOWN').astype(str)
        all_edges['is_mech_correct'] = all_edges['mechanism_true'] == all_edges['mechanism_pred']
        all_edges['is_gold_and_correct'] = all_edges['is_gold'] & all_edges['is_mech_correct']
        mvep_map = {}
        for k in MVEP_K_LIST:
            k_eff = min(k, len(all_edges))
            if k_eff == 0:
                mvep_map[k] = np.nan
            else:
                mvep_map[k] = float(all_edges['is_gold_and_correct'].iloc[:k_eff].sum() / k_eff)

        bems = 0.5 * edge_recall_k + 0.5 * macro_f1_mech

    # Benchmark-3: Two-Stage Overall Score (TSOS)
    stage1 = ods
    if np.isnan(edge_recall_k) or np.isnan(macro_f1_mech):
        stage2 = np.nan
    else:
        stage2 = 0.7 * macro_f1_mech + 0.3 * edge_recall_k
    tsos = 0.5 * stage1 + 0.5 * stage2 if not np.isnan(stage1) and not np.isnan(stage2) else np.nan

    summary = {
        'ods_recall_k': ods_recall_k,
        'ods_mrr': ods_mrr,
        'ods': ods,
        'bems_edge_recall_k': edge_recall_k,
        'bems_macro_f1_mech': macro_f1_mech,
        'bems': bems,
        'stage1': stage1,
        'stage2': stage2,
        'tsos': tsos
    }

    main_metrics = {
        'ev1_enrichment_at_k': enrichment_at_k,
        'ev2_recall_at_k': ods_recall_k,
        'ev3_mvep_at_k': mvep_map.get(RECALL_K, np.nan)
    }
    pd.DataFrame([main_metrics]).to_csv('external_validation_main_metrics.csv', index=False)

    si_metrics = {
        'ods_recall_k': ods_recall_k,
        'ods_mrr': ods_mrr,
        'ods': ods,
        'precision_at_k': precision_at_k,
        'enrichment_at_k': enrichment_at_k,
        'bems_edge_recall_k': edge_recall_k,
        'bems_macro_f1_mech': macro_f1_mech,
        'bems': bems,
        'stage1': stage1,
        'stage2': stage2,
        'tsos': tsos
    }
    for k, v in mvep_map.items():
        si_metrics[f'mvep_at_{k}'] = v
    pd.DataFrame([si_metrics]).to_csv('external_validation_si_metrics.csv', index=False)
    if not mech_cm.empty:
        mech_cm.to_csv('bn_benchmark_mechanism_confusion.csv')

    print("Saved main metrics to 'external_validation_main_metrics.csv'.")
    print("Saved SI metrics to 'external_validation_si_metrics.csv'.")
    if not mech_cm.empty:
        print("Saved mechanism confusion to 'bn_benchmark_mechanism_confusion.csv'.")

    # Figures
    if not down_df.empty:
        plot_down_rank_distribution(down_df, RECALL_K, enrichment_at_k, out_path='bn_benchmark_down_rank.png')
        if down_df['is_pos'].sum() > 0:
            plot_recall_at_k_curve(flags, total_pos, RECALL_K_LIST, out_path='bn_benchmark_recall_at_k.png')
    if not mech_cm.empty:
        plot_mechanism_confusion(mech_cm, out_path='bn_benchmark_mechanism_cm.png', normalize=True)
    if mvep_map:
        plot_mvep_at_k(mvep_map, out_path='bn_benchmark_mvep_at_k.png')
    plot_two_stage_summary(summary, out_path='bn_benchmark_two_stage.png')


if __name__ == '__main__':
    main()
