import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

EVAL_SETTINGS = {
    'max_parents_cap': 10,
    'pre': {
        'evidence_frac': 0.02,
        'top_k': 30,
        'hops': 2,
        'repeats': 3,
        'max_evidence_nodes': 50,
        'max_query_nodes': 400
    },
    'synthetic_calibration': {
        'n_samples': 80,
        'mask_frac': 0.3,
        'target_per_sample': 5,
        'n_bins': 10,
        'max_evidence_nodes': 60
    },
    'link_prediction': {
        'holdout_frac': 0.1,
        'negative_ratio': 1.0,
        'max_holdout_edges': 800
    },
    'robustness': {
        'drop_frac': 0.1,
        'base_k': 10,
        'alt_ks': (8, 12)
    }
}


def load_edges(edges_path='edges_info.csv'):
    edges_df = pd.read_csv(edges_path)
    return edges_df


def load_model_h5(filepath='bn_model.h5'):
    try:
        import h5py
    except Exception as e:
        raise ImportError("h5py is required to load H5 models. Install it first.") from e

    with h5py.File(filepath, 'r') as f:
        edges = [tuple(edge.astype(str)) for edge in f['edges'][()]]
        model = DiscreteBayesianNetwork(edges)

        for var in f['cpds'].keys():
            cpd_group = f['cpds'][var]
            values = cpd_group['values'][()]
            var_card = int(cpd_group['variable_card'][()][0])
            evidence = [e.decode('utf-8') if isinstance(e, bytes) else str(e) for e in cpd_group['evidence'][()]]
            evidence_card = [int(x) for x in cpd_group['evidence_card'][()]]
            cpd = TabularCPD(
                variable=var,
                variable_card=var_card,
                values=values,
                evidence=evidence if evidence else None,
                evidence_card=evidence_card if evidence_card else None
            )
            model.add_cpds(cpd)

    if not model.check_model():
        raise ValueError("Loaded model is invalid.")
    return model


def build_single_cpd(node, parents, edge_prob_map, max_parents=20):
    if not parents:
        cpd = TabularCPD(variable=node, variable_card=2, values=[[0.5], [0.5]])
        return cpd, []

    parent_influences = [(parent, edge_prob_map.get((parent, node), 0.5)) for parent in parents]
    parent_influences.sort(key=lambda x: x[1], reverse=True)
    selected_parents = [parent for parent, _ in parent_influences[:max_parents]]

    inhibitor_probs = [edge_prob_map.get((parent, node), 0.5) for parent in selected_parents]
    parent_states = list(itertools.product([0, 1], repeat=len(selected_parents)))

    values = []
    for state in parent_states:
        prob = 1.0
        for i, s in enumerate(state):
            if s == 1:
                prob *= (1 - inhibitor_probs[i])
        active_prob = 1 - prob
        values.append([1 - active_prob, active_prob])

    cpd = TabularCPD(
        variable=node,
        variable_card=2,
        values=list(map(list, zip(*values))),
        evidence=selected_parents,
        evidence_card=[2] * len(selected_parents)
    )
    return cpd, selected_parents


def build_model_from_edges(edges_df, edge_prob_map, max_parents=20, max_parents_cap=None):
    if max_parents_cap is not None:
        max_parents = min(max_parents, max_parents_cap)
    edges = list(zip(edges_df['Source'], edges_df['Target']))
    model = DiscreteBayesianNetwork(edges)

    cpds = []
    selected_parents_map = {}
    for node in model.nodes():
        parents = model.get_parents(node)
        cpd, selected_parents = build_single_cpd(node, parents, edge_prob_map, max_parents=max_parents)
        cpds.append(cpd)
        selected_parents_map[node] = selected_parents

    for node in model.nodes():
        if node in selected_parents_map:
            desired_parents = selected_parents_map[node]
            current_parents = model.get_parents(node)
            for parent in current_parents.copy():
                if parent not in desired_parents:
                    model.remove_edge(parent, node)

    model.add_cpds(*cpds)
    if not model.check_model():
        raise ValueError("Rebuilt Bayesian Network structure invalid!")
    return model


def compute_edge_fit(model, edges_df):
    infer = VariableElimination(model)
    results = []
    total = len(edges_df)
    for idx, row in edges_df.iterrows():
        parent = row['Source']
        child = row['Target']
        observed = float(row['CooccurrenceProbability'])
        try:
            q1 = infer.query(variables=[child], evidence={parent: 1})
            pred = float(q1.values[1])
            results.append((parent, child, observed, pred))
        except Exception:
            results.append((parent, child, observed, np.nan))
        if (idx + 1) % 100 == 0:
            print(f"Edge fit progress: {idx + 1}/{total}")

    fit_df = pd.DataFrame(results, columns=['Source', 'Target', 'Observed', 'Predicted'])
    valid = fit_df.dropna()
    if len(valid) > 1:
        mae = np.mean(np.abs(valid['Predicted'] - valid['Observed']))
        mse = np.mean((valid['Predicted'] - valid['Observed']) ** 2)
        pearson = np.corrcoef(valid['Predicted'], valid['Observed'])[0, 1]
        spearman = valid['Predicted'].rank().corr(valid['Observed'].rank())
    else:
        mae = mse = pearson = spearman = np.nan

    metrics = {
        'mae': mae,
        'mse': mse,
        'pearson': pearson,
        'spearman': spearman
    }
    return fit_df, metrics


def compute_marginals(model):
    infer = VariableElimination(model)
    probs = {}
    nodes = list(model.nodes())
    total = len(nodes)
    for idx, node in enumerate(nodes):
        try:
            q = infer.query(variables=[node])
            probs[node] = float(q.values[1])
        except Exception:
            probs[node] = np.nan
        if (idx + 1) % 50 == 0:
            print(f"Marginal progress: {idx + 1}/{total}")
    return probs


def compute_entropy(probs):
    entropies = {}
    for node, p in probs.items():
        if p is None or np.isnan(p):
            entropies[node] = np.nan
            continue
        p = min(max(p, 1e-12), 1 - 1e-12)
        ent = -p * np.log2(p) - (1 - p) * np.log2(1 - p)
        entropies[node] = float(ent)
    return entropies


def build_undirected_adj(edges_df):
    adj = {}
    for _, row in edges_df.iterrows():
        src = row['Source']
        tgt = row['Target']
        adj.setdefault(src, set()).add(tgt)
        adj.setdefault(tgt, set()).add(src)
    return adj


def get_k_hop_neighbors(adj, sources, hops=2):
    visited = set(sources)
    frontier = set(sources)
    for _ in range(hops):
        next_frontier = set()
        for node in frontier:
            next_frontier.update(adj.get(node, set()))
        next_frontier -= visited
        visited.update(next_frontier)
        frontier = next_frontier
    visited -= set(sources)
    return visited


def compute_pre(model, edges_df, evidence_frac=0.05, top_k=50, hops=2, repeats=5, max_evidence_nodes=None, max_query_nodes=None, seed=42):
    rng = np.random.default_rng(seed)
    nodes = list(model.nodes())
    adj = build_undirected_adj(edges_df)
    prior_probs = compute_marginals(model)
    infer = VariableElimination(model)

    rows = []
    for rep in range(repeats):
        evidence_size = max(1, int(len(nodes) * evidence_frac))
        if max_evidence_nodes is not None:
            evidence_size = min(evidence_size, max_evidence_nodes)
        evidence_nodes = rng.choice(nodes, size=evidence_size, replace=False)
        evidence_set = set(evidence_nodes)
        evidence = {node: 1 for node in evidence_set}

        query_nodes = [node for node in nodes if node not in evidence_set]
        if max_query_nodes is not None and len(query_nodes) > max_query_nodes:
            query_nodes = rng.choice(query_nodes, size=max_query_nodes, replace=False)

        deltas = {}
        for node in query_nodes:
            if node in evidence_set:
                continue
            prior = prior_probs.get(node, np.nan)
            if np.isnan(prior):
                continue
            try:
                q = infer.query(variables=[node], evidence=evidence)
                posterior = float(q.values[1])
            except Exception:
                continue
            deltas[node] = posterior - prior

        if not deltas:
            rows.append((rep, np.nan, np.nan, np.nan, np.nan))
            continue

        ranked = sorted(deltas.items(), key=lambda x: x[1], reverse=True)
        top_nodes = [node for node, _ in ranked[:min(top_k, len(ranked))]]
        neighborhood = get_k_hop_neighbors(adj, evidence_set, hops=hops)

        observed_frac = len([node for node in top_nodes if node in neighborhood]) / max(len(top_nodes), 1)
        baseline_fracs = []
        candidate_pool = list(deltas.keys())
        for _ in range(20):
            sample_nodes = rng.choice(candidate_pool, size=len(top_nodes), replace=False)
            baseline_fracs.append(len([node for node in sample_nodes if node in neighborhood]) / max(len(top_nodes), 1))
        expected_frac = float(np.mean(baseline_fracs)) if baseline_fracs else np.nan
        enrichment = observed_frac / expected_frac if expected_frac and expected_frac > 0 else np.nan

        neigh_deltas = [deltas[node] for node in deltas if node in neighborhood]
        non_deltas = [deltas[node] for node in deltas if node not in neighborhood]
        mean_neigh = float(np.mean(neigh_deltas)) if neigh_deltas else np.nan
        mean_non = float(np.mean(non_deltas)) if non_deltas else np.nan
        effect = mean_neigh - mean_non if not np.isnan(mean_neigh) and not np.isnan(mean_non) else np.nan

        rows.append((rep, enrichment, mean_neigh, mean_non, effect))

    pre_df = pd.DataFrame(rows, columns=['Repeat', 'Enrichment', 'MeanDeltaNeighborhood', 'MeanDeltaNonNeighborhood', 'DeltaEffect'])
    pre_df.to_csv('bn_model_pre.csv', index=False)
    print("Saved PRE report to 'bn_model_pre.csv'.")
    return pre_df


def perturb_edge_probs(edges_df, scale=0.1, seed=42):
    rng = np.random.default_rng(seed)
    probs = edges_df['CooccurrenceProbability'].to_numpy(dtype=float)
    noise = rng.uniform(-scale, scale, size=probs.shape)
    perturbed = np.clip(probs * (1 + noise), 0.0, 1.0)
    return perturbed


def plot_hist(data, title, xlabel, out_path):
    plt.figure(figsize=(8, 6))
    plt.hist(data, bins=30, color='#4C72B0', alpha=0.85)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

def compute_ece(y_true, y_score, n_bins=10):
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_score, bins, right=True)
    rows = []
    ece = 0.0
    total = len(y_true)

    for b in range(1, n_bins + 1):
        mask = bin_ids == b
        if not np.any(mask):
            rows.append((b, np.nan, np.nan, 0))
            continue
        avg_pred = float(np.mean(y_score[mask]))
        avg_true = float(np.mean(y_true[mask]))
        count = int(np.sum(mask))
        ece += abs(avg_pred - avg_true) * (count / max(total, 1))
        rows.append((b, avg_pred, avg_true, count))

    ece_df = pd.DataFrame(rows, columns=['Bin', 'AvgPred', 'AvgTrue', 'Count'])
    return float(ece), ece_df


def plot_reliability(ece_df, out_path):
    fig, ax = plt.subplots(figsize=(6.5, 6))
    valid = ece_df.dropna()
    ax.plot([0, 1], [0, 1], color='#999999', linestyle='--', linewidth=1, label='Perfect Calibration')
    ax.plot(valid['AvgPred'], valid['AvgTrue'], color='#377EB8', marker='o', linewidth=2, label='BN')
    ax.set_xlabel('Mean Predicted Probability')
    ax.set_ylabel('Mean True Frequency')
    ax.set_title('Reliability Diagram (Synthetic Sampling)')

    ax2 = ax.twinx()
    ax2.bar(valid['AvgPred'], valid['Count'], width=0.06, color='#BBBBBB', alpha=0.4, label='Bin Count')
    ax2.set_ylabel('Count')
    ax2.tick_params(axis='y', labelcolor='#666666')

    ax.legend(loc='lower right')
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

def compute_pr_auc(y_true, y_score):
    thresholds = np.unique(y_score)[::-1]
    precision_list = []
    recall_list = []
    pos_total = (y_true == 1).sum()

    for thr in thresholds:
        y_pred = (y_score >= thr).astype(int)
        tp = ((y_true == 1) & (y_pred == 1)).sum()
        fp = ((y_true == 0) & (y_pred == 1)).sum()
        fn = ((y_true == 1) & (y_pred == 0)).sum()

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        precision_list.append(precision)
        recall_list.append(recall)

    precision_list = [1.0] + precision_list
    recall_list = [0.0] + recall_list
    if pos_total > 0 and recall_list[-1] < 1.0:
        precision_list.append(pos_total / max(len(y_true), 1))
        recall_list.append(1.0)

    pr_auc = np.trapezoid(precision_list, recall_list)
    return np.array(recall_list), np.array(precision_list), pr_auc

def compute_brier_score(y_true, y_score):
    return float(np.mean((y_score - y_true) ** 2))

def plot_pr(recall, precision, pr_auc, baseline_precision, out_path):
    plt.figure(figsize=(6, 6))
    plt.plot(recall, precision, color='#1B9E77', linewidth=2, label=f'BN (Δ-influence) PR-AUC = {pr_auc:.3f}')
    plt.plot([0, 1], [baseline_precision, baseline_precision], color='#888888', linestyle='--', linewidth=1, label='Random Baseline')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall (Δ-influence Link Prediction)')
    plt.legend(loc='lower left')
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

def forward_sample(model, n_samples=200, seed=42):
    try:
        from pgmpy.sampling import BayesianModelSampling
    except Exception as e:
        raise ImportError("pgmpy.sampling.BayesianModelSampling is required for synthetic calibration.") from e

    sampler = BayesianModelSampling(model)
    samples = sampler.forward_sample(size=n_samples, seed=seed)
    return samples


def compute_synthetic_calibration(model, n_samples=200, mask_frac=0.3, target_per_sample=10, n_bins=10, max_evidence_nodes=None, seed=42):
    rng = np.random.default_rng(seed)
    samples = forward_sample(model, n_samples=n_samples, seed=seed)
    nodes = list(model.nodes())
    infer = VariableElimination(model)

    rows = []
    for idx in range(len(samples)):
        row = samples.iloc[idx]
        observed_size = max(1, int(len(nodes) * (1 - mask_frac)))
        if max_evidence_nodes is not None:
            observed_size = min(observed_size, max_evidence_nodes)
        observed_nodes = rng.choice(nodes, size=observed_size, replace=False)
        evidence = {node: int(row[node]) for node in observed_nodes}

        target_nodes = [node for node in nodes if node not in evidence]
        if target_per_sample and len(target_nodes) > target_per_sample:
            target_nodes = rng.choice(target_nodes, size=target_per_sample, replace=False)

        for node in target_nodes:
            try:
                q = infer.query(variables=[node], evidence=evidence)
                prob = float(q.values[1])
            except Exception:
                prob = np.nan
            rows.append((node, int(row[node]), prob))

    synth_df = pd.DataFrame(rows, columns=['Node', 'TrueState', 'PredProb'])
    synth_df.to_csv('bn_model_synth_calibration.csv', index=False)
    print("Saved synthetic calibration table to 'bn_model_synth_calibration.csv'.")

    valid = synth_df.dropna()
    if len(valid) == 0:
        return np.nan, np.nan

    y_true = valid['TrueState'].to_numpy(dtype=int)
    y_score = valid['PredProb'].to_numpy(dtype=float)
    brier = compute_brier_score(y_true, y_score)
    ece, ece_df = compute_ece(y_true, y_score, n_bins=n_bins)
    ece_df.to_csv('bn_model_synth_reliability.csv', index=False)
    plot_reliability(ece_df, out_path='bn_model_synth_reliability.png')
    print("Saved synthetic reliability to 'bn_model_synth_reliability.csv' and 'bn_model_synth_reliability.png'.")

    return brier, ece

def compute_variability(nodes_path='nodes_info_with_inference100.csv'):
    df = pd.read_csv(nodes_path)
    required = {'Node', 'bayes_probability_inferred', 'BaseProbability'}
    if not required.issubset(df.columns):
        raise ValueError("nodes_info_with_inference100.csv must contain Node, BaseProbability, bayes_probability_inferred.")

    probs = df['bayes_probability_inferred'].astype(float)
    delta = probs - df['BaseProbability'].astype(float)

    variability = {
        'mean_prob': float(probs.mean()),
        'std_prob': float(probs.std(ddof=0)),
        'iqr_prob': float(probs.quantile(0.75) - probs.quantile(0.25)),
        'mean_delta': float(delta.mean()),
        'std_delta': float(delta.std(ddof=0)),
        'iqr_delta': float(delta.quantile(0.75) - delta.quantile(0.25))
    }
    variability_df = pd.DataFrame([variability])
    variability_df.to_csv('bn_model_variability.csv', index=False)
    print("Saved variability report to 'bn_model_variability.csv'.")

    plot_hist(probs.dropna(), 'Posterior Probability Distribution', 'P(node=1)', 'bn_model_prob_hist.png')
    plot_hist(delta.dropna(), 'Posterior Delta Distribution', 'P(node=1) - BaseProbability', 'bn_model_delta_hist.png')
    print("Saved probability histograms to 'bn_model_prob_hist.png' and 'bn_model_delta_hist.png'.")

def compute_delta_influence(infer, parent, child):
    q1 = infer.query(variables=[child], evidence={parent: 1})
    q0 = infer.query(variables=[child], evidence={parent: 0})
    return float(q1.values[1] - q0.values[1])


def build_two_hop_map(adj):
    two_hop = {}
    for node in adj:
        one_hop = adj.get(node, set())
        two_hop_nodes = set(one_hop)
        for neigh in one_hop:
            two_hop_nodes.update(adj.get(neigh, set()))
        two_hop_nodes.discard(node)
        two_hop[node] = two_hop_nodes
    return two_hop


def compute_link_prediction(model, edges_df, holdout_frac=0.1, negative_ratio=1.0, max_holdout_edges=None, max_parents_cap=None, seed=42):
    rng = np.random.default_rng(seed)
    edges_df = edges_df.copy()
    edges_df['edge_id'] = np.arange(len(edges_df))
    holdout_size = max(1, int(len(edges_df) * holdout_frac))
    holdout_ids = rng.choice(edges_df['edge_id'], size=holdout_size, replace=False)

    holdout_df = edges_df[edges_df['edge_id'].isin(holdout_ids)]
    if max_holdout_edges is not None and len(holdout_df) > max_holdout_edges:
        holdout_df = holdout_df.sample(n=max_holdout_edges, random_state=seed)
    train_df = edges_df[~edges_df['edge_id'].isin(holdout_ids)].drop(columns=['edge_id'])

    edge_prob_map = {(row['Source'], row['Target']): float(row['CooccurrenceProbability'])
                     for _, row in train_df.iterrows()}
    train_model = build_model_from_edges(train_df, edge_prob_map, max_parents=20, max_parents_cap=max_parents_cap)
    infer = VariableElimination(train_model)

    nodes = sorted(set(edges_df['Source']).union(set(edges_df['Target'])))
    edge_set = set(zip(edges_df['Source'], edges_df['Target']))
    adj = build_undirected_adj(edges_df)
    two_hop_map = build_two_hop_map(adj)

    neg_needed = int(len(holdout_df) * negative_ratio)
    neg_pairs = set()
    holdout_parents = holdout_df['Source'].tolist()
    attempts = 0
    while len(neg_pairs) < neg_needed and attempts < neg_needed * 20:
        attempts += 1
        u = rng.choice(holdout_parents)
        candidates = list(two_hop_map.get(u, set()))
        rng.shuffle(candidates)
        v = None
        for cand in candidates:
            if cand == u or (u, cand) in edge_set:
                continue
            v = cand
            break
        if v is None:
            v = rng.choice(nodes)
            if v == u or (u, v) in edge_set:
                continue
        neg_pairs.add((u, v))

    rows = []
    for _, row in holdout_df.iterrows():
        parent = row['Source']
        child = row['Target']
        try:
            score = compute_delta_influence(infer, parent, child)
        except Exception:
            score = np.nan
        rows.append((parent, child, 1, score))

    for parent, child in neg_pairs:
        try:
            score = compute_delta_influence(infer, parent, child)
        except Exception:
            score = np.nan
        rows.append((parent, child, 0, score))

    lp_df = pd.DataFrame(rows, columns=['Source', 'Target', 'Label', 'Score'])
    lp_df.to_csv('bn_model_link_pred.csv', index=False)
    print("Saved link prediction scores to 'bn_model_link_pred.csv'.")

    valid = lp_df.dropna()
    y_true = valid['Label'].to_numpy(dtype=int)
    y_score = valid['Score'].to_numpy(dtype=float)
    if len(valid) == 0:
        pr_auc = np.nan
    else:
        recall, precision, pr_auc = compute_pr_auc(y_true, y_score)
        pr_df = pd.DataFrame({'recall': recall, 'precision': precision})
        pr_df.to_csv('bn_model_link_pred_pr.csv', index=False)
        baseline_precision = float(np.mean(y_true))
        plot_pr(recall, precision, pr_auc, baseline_precision, out_path='bn_model_link_pred_pr.png')
        print("Saved link prediction PR curve to 'bn_model_link_pred_pr.csv' and 'bn_model_link_pred_pr.png'.")

    return pr_auc


def build_structure_perturbed_edges(edges_df, drop_frac=0.1):
    groups = []
    for target, group in edges_df.groupby('Target'):
        group = group.sort_values('CooccurrenceProbability', ascending=True)
        drop_n = int(len(group) * drop_frac)
        if drop_n > 0:
            group = group.iloc[drop_n:]
        groups.append(group)
    return pd.concat(groups, ignore_index=True)


def kendall_tau(values_a, values_b, sample_size=300, seed=42):
    if len(values_a) != len(values_b):
        raise ValueError("kendall_tau requires arrays of the same length.")
    n = len(values_a)
    if n < 2:
        return np.nan

    rng = np.random.default_rng(seed)
    if n > sample_size:
        idx = rng.choice(n, size=sample_size, replace=False)
        values_a = values_a[idx]
        values_b = values_b[idx]
        n = len(values_a)

    concordant = 0
    discordant = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            dx = values_a[i] - values_a[j]
            dy = values_b[i] - values_b[j]
            if dx == 0 or dy == 0:
                continue
            if dx * dy > 0:
                concordant += 1
            else:
                discordant += 1

    denom = concordant + discordant
    return (concordant - discordant) / denom if denom > 0 else np.nan


def compute_topk_sensitivity(edges_df, edge_prob_map, base_k=20, alt_ks=(10, 30), max_parents_cap=None, seed=42):
    base_model = build_model_from_edges(edges_df, edge_prob_map, max_parents=base_k, max_parents_cap=max_parents_cap)
    base_probs = compute_marginals(base_model)
    nodes = sorted(base_probs.keys())
    base_values = np.array([base_probs.get(node, np.nan) for node in nodes], dtype=float)

    results = []
    for k in alt_ks:
        alt_model = build_model_from_edges(edges_df, edge_prob_map, max_parents=k, max_parents_cap=max_parents_cap)
        alt_probs = compute_marginals(alt_model)
        alt_values = np.array([alt_probs.get(node, np.nan) for node in nodes], dtype=float)

        mask = ~np.isnan(base_values) & ~np.isnan(alt_values)
        tau = kendall_tau(base_values[mask], alt_values[mask], seed=seed)
        results.append((k, tau))

    sensitivity_df = pd.DataFrame(results, columns=['MaxParents', 'KendallTau'])
    sensitivity_df.to_csv('bn_model_topk_sensitivity.csv', index=False)
    print("Saved top-k sensitivity to 'bn_model_topk_sensitivity.csv'.")
    return sensitivity_df

def summarize_results(pre_df, synth_brier, synth_ece, pr_auc, structure_df, sensitivity_df, out_path='bn_model_summary.txt'):
    pre_valid = pre_df.dropna()
    if pre_valid.empty:
        pre_enrichment = np.nan
        pre_effect = np.nan
        pre_mean_neigh = np.nan
        pre_mean_non = np.nan
    else:
        pre_enrichment = float(pre_valid['Enrichment'].mean())
        pre_effect = float(pre_valid['DeltaEffect'].mean())
        pre_mean_neigh = float(pre_valid['MeanDeltaNeighborhood'].mean())
        pre_mean_non = float(pre_valid['MeanDeltaNonNeighborhood'].mean())

    valid_diffs = structure_df['AbsDiff'].dropna()
    if valid_diffs.empty:
        diff_median = np.nan
        diff_p95 = np.nan
    else:
        diff_median = float(valid_diffs.median())
        diff_p95 = float(valid_diffs.quantile(0.95))

    tau10 = np.nan
    tau30 = np.nan
    if not sensitivity_df.empty:
        for _, row in sensitivity_df.iterrows():
            if int(row['MaxParents']) == 10:
                tau10 = float(row['KendallTau'])
            if int(row['MaxParents']) == 30:
                tau30 = float(row['KendallTau'])

    lines = [
        "BN model summary",
        f"- PRE_enrichment_at_K: {pre_enrichment:.3f}x" if not np.isnan(pre_enrichment) else "- PRE_enrichment_at_K: nan",
        f"- PRE_delta_effect: {pre_effect:.4f}" if not np.isnan(pre_effect) else "- PRE_delta_effect: nan",
        f"- PRE_delta_mean_neighborhood: {pre_mean_neigh:.4f}" if not np.isnan(pre_mean_neigh) else "- PRE_delta_mean_neighborhood: nan",
        f"- PRE_delta_mean_non_neighborhood: {pre_mean_non:.4f}" if not np.isnan(pre_mean_non) else "- PRE_delta_mean_non_neighborhood: nan",
        f"- Synthetic_calibration_Brier: {synth_brier:.4f}" if not np.isnan(synth_brier) else "- Synthetic_calibration_Brier: nan",
        f"- Synthetic_calibration_ECE: {synth_ece:.4f}" if not np.isnan(synth_ece) else "- Synthetic_calibration_ECE: nan",
        f"- Holdout_PR_AUC_delta_influence: {pr_auc:.4f}" if not np.isnan(pr_auc) else "- Holdout_PR_AUC_delta_influence: nan",
        f"- Robustness_structure_absdiff_median: {diff_median:.4f}" if not np.isnan(diff_median) else "- Robustness_structure_absdiff_median: nan",
        f"- Robustness_structure_absdiff_p95: {diff_p95:.4f}" if not np.isnan(diff_p95) else "- Robustness_structure_absdiff_p95: nan",
        f"- Robustness_topk_sensitivity_tau_10: {tau10:.4f}" if not np.isnan(tau10) else "- Robustness_topk_sensitivity_tau_10: nan",
        f"- Robustness_topk_sensitivity_tau_30: {tau30:.4f}" if not np.isnan(tau30) else "- Robustness_topk_sensitivity_tau_30: nan"
    ]
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write("\\n".join(lines) + "\\n")
    print(f"Saved summary to '{out_path}'.")


def main():
    start = time.time()

    if not os.path.exists('bn_model.h5'):
        raise FileNotFoundError("bn_model.h5 not found. Run Step3-BN_model.py first.")

    edges_df = load_edges()
    model = load_model_h5('bn_model.h5')

    # 1) Posterior Recovery under Evidence (PRE)
    pre_cfg = EVAL_SETTINGS['pre']
    pre_df = compute_pre(
        model,
        edges_df,
        evidence_frac=pre_cfg['evidence_frac'],
        top_k=pre_cfg['top_k'],
        hops=pre_cfg['hops'],
        repeats=pre_cfg['repeats'],
        max_evidence_nodes=pre_cfg['max_evidence_nodes'],
        max_query_nodes=pre_cfg['max_query_nodes'],
        seed=42
    )

    # 2) Synthetic calibration under controlled sampling
    synth_cfg = EVAL_SETTINGS['synthetic_calibration']
    synth_brier, synth_ece = compute_synthetic_calibration(
        model,
        n_samples=synth_cfg['n_samples'],
        mask_frac=synth_cfg['mask_frac'],
        target_per_sample=synth_cfg['target_per_sample'],
        n_bins=synth_cfg['n_bins'],
        max_evidence_nodes=synth_cfg['max_evidence_nodes'],
        seed=42
    )
    print(f"Synthetic calibration Brier: {synth_brier:.4f}")
    print(f"Synthetic calibration ECE: {synth_ece:.4f}")

    # 3) Holdout link prediction with delta influence score and hard negatives
    lp_cfg = EVAL_SETTINGS['link_prediction']
    pr_auc = compute_link_prediction(
        model,
        edges_df,
        holdout_frac=lp_cfg['holdout_frac'],
        negative_ratio=lp_cfg['negative_ratio'],
        max_holdout_edges=lp_cfg['max_holdout_edges'],
        max_parents_cap=EVAL_SETTINGS['max_parents_cap'],
        seed=42
    )
    print(f"Holdout link prediction PR-AUC (delta influence): {pr_auc:.4f}")

    # 4) Robustness under structural perturbation
    robust_cfg = EVAL_SETTINGS['robustness']
    perturbed_edges = build_structure_perturbed_edges(edges_df, drop_frac=robust_cfg['drop_frac'])
    perturbed_map = {(row['Source'], row['Target']): float(row['CooccurrenceProbability'])
                     for _, row in perturbed_edges.iterrows()}
    perturbed_model = build_model_from_edges(
        perturbed_edges,
        perturbed_map,
        max_parents=20,
        max_parents_cap=EVAL_SETTINGS['max_parents_cap']
    )
    base_probs = compute_marginals(model)
    perturbed_probs_map = compute_marginals(perturbed_model)

    structure_rows = []
    for node in model.nodes():
        base_p = base_probs.get(node, np.nan)
        pert_p = perturbed_probs_map.get(node, np.nan)
        abs_diff = np.nan
        if not np.isnan(base_p) and not np.isnan(pert_p):
            abs_diff = abs(base_p - pert_p)
        structure_rows.append((node, base_p, pert_p, abs_diff))

    structure_df = pd.DataFrame(structure_rows, columns=['Node', 'BaseProb', 'PerturbedProb', 'AbsDiff'])
    structure_df.to_csv('bn_model_structure_robustness.csv', index=False)
    print("Saved structure robustness report to 'bn_model_structure_robustness.csv'.")

    valid_diffs = structure_df['AbsDiff'].dropna()
    if not valid_diffs.empty:
        plot_hist(valid_diffs, 'Robustness |AbsDiff| under Structure Perturbation', 'AbsDiff', 'bn_model_structure_robustness_hist.png')
        print("Saved structure robustness histogram to 'bn_model_structure_robustness_hist.png'.")

    # 5) Top-k sensitivity for parent truncation
    edge_prob_map = {(row['Source'], row['Target']): float(row['CooccurrenceProbability'])
                     for _, row in edges_df.iterrows()}
    sensitivity_df = compute_topk_sensitivity(
        edges_df,
        edge_prob_map,
        base_k=robust_cfg['base_k'],
        alt_ks=robust_cfg['alt_ks'],
        max_parents_cap=EVAL_SETTINGS['max_parents_cap'],
        seed=42
    )

    summarize_results(pre_df, synth_brier, synth_ece, pr_auc, structure_df, sensitivity_df)

    print(f"Done. Total time: {time.time() - start:.2f}s")


if __name__ == '__main__':
    main()
