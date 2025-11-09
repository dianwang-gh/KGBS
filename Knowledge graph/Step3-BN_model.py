import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import time
import threading
import concurrent.futures
import psutil
import numpy as np
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
from collections import defaultdict
import itertools
import matplotlib.patches as mpatches

global_status = {'phase': 'initializing', 'start_time': time.time()}

# ------------------------ Inference Function ------------------------
def infer_node(model, node):
    """Execute probability inference for a single node"""
    try:
        infer = VariableElimination(model)
        q = infer.query(variables=[node])
        return (node, q.values[1])  # Return probability of active state
    except Exception as e:
        print(f"Error inferring {node}: {str(e)}")
        return (node, 0.5)  # Return default value

# ------------------------ Step 0: Start Countdown Thread ------------------------
def countdown_monitor():
    while True:
        elapsed = time.time() - global_status['start_time']
        print(f"[Timer] Phase: {global_status['phase']} - Elapsed: {int(elapsed)} seconds")
        time.sleep(30)

# ------------------------ Step 1: Load Node and Edge Information ------------------------
def load_data(nodes_path='nodes_info.csv', edges_path='edges_info.csv'):
    global_status['phase'] = 'Loading Data'
    start = time.time()
    nodes_df = pd.read_csv(nodes_path)
    edges_df = pd.read_csv(edges_path)
    print(f"Loaded {len(nodes_df)} nodes and {len(edges_df)} edges. Time: {time.time() - start:.2f}s")
    return nodes_df, edges_df

# ------------------------ Step 2: Build BayesianNetwork Structure ------------------------
def build_bayesian_model(edges_df):
    global_status['phase'] = 'Building Model'
    start = time.time()
    edges = list(zip(edges_df['Source'], edges_df['Target']))
    model = DiscreteBayesianNetwork(edges)
    print(f"Bayesian model built with {len(edges)} edges. Time: {time.time() - start:.2f}s")
    return model

# ------------------------ Step 3: Single Node CPD Construction Function ------------------------
def build_single_cpd(node, parents, edge_prob_map, threshold=0.01):
    try:
        if not parents:
            cpd = TabularCPD(variable=node, variable_card=2, values=[[0.5], [0.5]])
            return cpd, []

        # Filter parent nodes with significant influence
        parent_influences = [(parent, edge_prob_map.get((parent, node), 0.5)) for parent in parents]
        # Sort by influence degree and select top 20
        parent_influences.sort(key=lambda x: x[1], reverse=True)
        selected_parents = [parent for parent, _ in parent_influences[:20]]
        # This number mainly targets mCPP node with 91 parent nodes, too many parent nodes cannot produce results
        print(f"Simplified {node}: Reduced parents from {len(parents)} to {len(selected_parents)}")

        # Rebuild CPD using filtered parent nodes
        inhibitor_probs = [edge_prob_map.get((parent, node), 0.5) for parent in selected_parents]

        # Generate all parent node state combinations
        parent_states = list(itertools.product([0, 1], repeat=len(selected_parents)))

        values = []
        for state in parent_states:
            # Probability = 1 - ∏(1 - p_i * x_i)
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
    except Exception as e:
        print(f"Error building simplified CPD for {node}: {str(e)}")
        raise

# ------------------------ Step 4: Optimized Parallel CPD Creation ------------------------
def create_cpds_parallel(model, edges_df):
    global_status['phase'] = 'Creating CPDs'
    start = time.time()
    process = psutil.Process()

    edge_prob_map = {(row['Source'], row['Target']): row['CooccurrenceProbability']
                     for _, row in edges_df.iterrows()}

    node_groups = defaultdict(list)
    for node in model.nodes():
        parents = model.get_parents(node)
        node_groups[len(parents)].append((node, parents))

    cpds = []
    selected_parents_map = {}
    for complexity in sorted(node_groups.keys()):
        group = node_groups[complexity]
        print(f"Processing {len(group)} nodes with {complexity} parents...")
        with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(build_single_cpd, node, parents, edge_prob_map): node for (node, parents) in group}
            for future in concurrent.futures.as_completed(futures):
                node = futures[future]
                try:
                    cpd, selected_parents = future.result()
                    cpds.append(cpd)
                    selected_parents_map[node] = selected_parents
                    if len(cpds) % 10 == 0:
                        mem_usage = process.memory_info().rss // 1024 ** 2
                        print(f"Memory usage: {mem_usage}MB | Progress: {len(cpds)}/{len(model.nodes())}")
                except Exception as e:
                    print(f"!!! Critical error: {e}")
                    raise

    print(f"Created {len(cpds)} CPDs. Total time: {time.time() - start:.2f}s")
    return cpds, selected_parents_map


# ------------------------ Step 5: Optimized Inference Logic ------------------------
def infer_bayes_probabilities_parallel(model, nodes_df):
	global_status['phase'] = 'Inference'
	start = time.time()
	total_nodes = len(model.nodes())

	# Batch inference
	BATCH_SIZE = 50
	nodes = list(model.nodes())
	bayes_probs = []

	# Correct CPD serialization method
	model_data = {
		'edges': list(model.edges()),
		'cpds': [{
			'variable': cpd.variable,
			'evidence': cpd.variables[1:],  # variables[0] is current node, followed by parent nodes
			'values': cpd.get_values().tolist(),
			'evidence_card': cpd.cardinality[1:]
		} for cpd in model.cpds]
	}

	for i in range(0, len(nodes), BATCH_SIZE):
		batch = nodes[i:i + BATCH_SIZE]
		with concurrent.futures.ProcessPoolExecutor(max_workers=2) as executor:
			futures = {executor.submit(_infer_wrapper, model_data, node): node for node in batch}

			for future in concurrent.futures.as_completed(futures):
				node = futures[future]
				try:
					prob = future.result()
					bayes_probs.append((node, prob))
					print(f"Progress: {len(bayes_probs)}/{total_nodes}")
				except Exception as e:
					print(f"Error processing {node}: {e}")

	nodes_df['bayes_probability_inferred'] = nodes_df['Node'].map(dict(bayes_probs))
	print(f"Inference completed. Time: {time.time() - start:.2f}s")
	return nodes_df


# ------------------------ Wrapper Function ------------------------
def _infer_wrapper(model_data, node):
	"""Rebuild model and execute inference"""
	try:
		# Rebuild network structure
		model = DiscreteBayesianNetwork(model_data['edges'])

		# Rebuild CPDs
		for cpd_data in model_data['cpds']:
			cpd = TabularCPD(
				variable=cpd_data['variable'],
				variable_card=2,
				values=cpd_data['values'],
				evidence=cpd_data['evidence'],
				evidence_card=cpd_data['evidence_card']
			)
			model.add_cpds(cpd)

		# Execute inference
		infer = VariableElimination(model)
		q = infer.query(variables=[node])
		return q.values[1]
	except Exception as e:
		print(f"Inference failed for {node}: {str(e)}")
		return 0.5


# ------------------------ Step 6: Visualization Function ------------------------
def visualize_inferred_network(nodes_df, edges_df, save_path='inferred_network.png'):
	global_status['phase'] = 'Visualization'
	start = time.time()
	G = nx.DiGraph()
	for _, row in edges_df.iterrows():
		G.add_edge(row['Source'], row['Target'])

	if 'Type' not in nodes_df.columns:
		raise ValueError("Nodes DataFrame must contain 'Type' column")

	node_type_map = nodes_df.set_index('Node')['Type'].to_dict()
	bayes_normalized = {row['Node']: row['bayes_normalized'] for _, row in nodes_df.iterrows()}

	plt.figure(figsize=(33, 30)) #35 30
	ax = plt.gca()
	pos = nx.spring_layout(G, k=0.7, iterations=50)

	# Node size mapping
	node_size_map = {}
	for node in G.nodes():
		if node_type_map.get(node) == 'PAEs':
			node_size_map[node] = 3000
		else:
			norm_val = bayes_normalized.get(node, 50)
			scaled_val = np.sqrt(norm_val)
			node_size_map[node] = 2500 + 30 * scaled_val * 10

	node_sizes_list = [node_size_map[node] for node in G.nodes()]
	nx.draw_networkx_edges(
		G, pos, ax=ax,
		edge_color='#d9d9d9',
		arrows=True,
		arrowstyle='-|>,head_length=0.4,head_width=0.2',
		arrowsize=15,
		connectionstyle='arc3,rad=0.1',
		alpha=0.8,
		width=1.5,
		node_size=node_sizes_list
	)

	PAEs_nodes = [node for node in G.nodes() if node_type_map.get(node) == 'PAEs']
	mPAEs_nodes = [node for node in G.nodes() if node_type_map.get(node) == 'mPAEs']

	if PAEs_nodes:
		PAEs_sizes = [node_size_map[node] for node in PAEs_nodes]
		PAEs_pos = {node: pos[node] for node in PAEs_nodes}
		nx.draw_networkx_nodes(
			G, PAEs_pos,
			nodelist=PAEs_nodes,
			node_size=PAEs_sizes,
			node_color='#d6d6d6',
			edgecolors='#bcbcbc',
			alpha=0.8,
			linewidths=3,
			ax=ax
		)

	if mPAEs_nodes:
		mPAEs_sizes = [node_size_map[node] for node in mPAEs_nodes]
		mPAEs_normalized = [bayes_normalized.get(node, 50) for node in mPAEs_nodes]
		mPAEs_pos = {node: pos[node] for node in mPAEs_nodes}

		# Define classification boundaries and corresponding colors
		bins = [0, 0.01, 0.1, 1, 20, 100]
		colors = [
			'#F6ECD5',  # <0.01
			'#F9CA71',  # 0.01-0.1
			'#E87630',  # 0.1-1
			'#B3344E',  # 1-20
			'#7E1A6A'  # 20-100
		]

		# Assign color for each node
		mPAEs_colors = []
		for val in mPAEs_normalized:
			if val < bins[1]:
				mPAEs_colors.append(colors[0])
			elif val < bins[2]:
				mPAEs_colors.append(colors[1])
			elif val < bins[3]:
				mPAEs_colors.append(colors[2])
			elif val < bins[4]:
				mPAEs_colors.append(colors[3])
			else:
				mPAEs_colors.append(colors[4])

		# Draw nodes (using discrete colors)
		nx.draw_networkx_nodes(
			G, mPAEs_pos,
			nodelist=mPAEs_nodes,
			node_size=mPAEs_sizes,
			node_color=mPAEs_colors,
			edgecolors='#bcbcbc',
			linewidths=3,
			alpha=0.8,
			ax=ax
		)

	nx.draw_networkx_labels(G, pos, font_size=20, ax=ax, font_family='Arial')

	# Create custom legend
	legend_patches = [
		mpatches.Patch(color='#d6d6d6', label='PAEs'),
		mpatches.Patch(color=colors[0], label='mPAEs: <0.01'),
		mpatches.Patch(color=colors[1], label='mPAEs: 0.01-0.1'),
		mpatches.Patch(color=colors[2], label='mPAEs: 0.1-1'),
		mpatches.Patch(color=colors[3], label='mPAEs: 1-20'),
		mpatches.Patch(color=colors[4], label='mPAEs: 20-100')
	]

	legend = ax.legend(
		handles=legend_patches,
		loc='upper right',
		bbox_to_anchor=(1.1, 1),
		frameon=False,
		framealpha=0.8,
		edgecolor='#333333',
		facecolor='white',
		fontsize=25,
		borderpad=1.0,
		handlelength=1.2,
		handleheight=1.2,
		handletextpad=0.4,
		labelspacing=0.4,
		borderaxespad=0.6
	)

	plt.axis('off')
	plt.tight_layout()
	plt.savefig(save_path, format='PNG', dpi=300, bbox_inches='tight')
	plt.close()
	print(f"Visualization saved as '{save_path}'. Total time: {time.time() - start:.2f}s")

# ------------------------ Main Program ------------------------
def main():
    monitor_thread = threading.Thread(target=countdown_monitor, daemon=True)
    monitor_thread.start()

    nodes_df, edges_df = load_data()

    model = build_bayesian_model(edges_df)

    cpds, selected_parents_map = create_cpds_parallel(model, edges_df)

    # Adjust model structure, remove unselected parent nodes
    for node in model.nodes():
        if node in selected_parents_map:
            desired_parents = selected_parents_map[node]
            current_parents = model.get_parents(node)
            for parent in current_parents.copy():
                if parent not in desired_parents:
                    model.remove_edge(parent, node)

    model.add_cpds(*cpds)

    if not model.check_model():
        raise ValueError("Bayesian Network structure invalid!")

    updated_nodes_df = infer_bayes_probabilities_parallel(model, nodes_df)
    # Only normalize mPAEs nodes (0-100)
    # Separate mPAEs nodes
    mpaes_mask = updated_nodes_df['Type'] == 'mPAEs'
    mpaes_probs = updated_nodes_df.loc[mpaes_mask, 'bayes_probability_inferred']

    # Calculate min and max probabilities for mPAEs
    min_prob = mpaes_probs.min()
    max_prob = mpaes_probs.max()

    # Apply normalization only to mPAEs nodes
    updated_nodes_df.loc[mpaes_mask, 'bayes_normalized'] = 100 * (
			    updated_nodes_df.loc[mpaes_mask, 'bayes_probability_inferred'] - min_prob) / (max_prob - min_prob)

    # PAEs nodes set to fixed value 50 (middle value, doesn't affect color mapping)
    updated_nodes_df.loc[~mpaes_mask, 'bayes_normalized'] = 50

    updated_nodes_df.to_csv('nodes_info_with_inference100.csv', index=False)
    print("Saved updated node probabilities to 'nodes_info_with_inference100.csv'.")

    visualize_inferred_network(updated_nodes_df, edges_df)

if __name__ == "__main__":
    main()