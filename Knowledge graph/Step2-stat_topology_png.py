import networkx as nx
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np


# ------------------------ Step 1: Load Graph and Build Mechanism Attributes ------------------------
def load_graph(filepath='PAEs_network.gpickle', tpr_filepath='TPR_number.csv'):
	# Load graph structure
	with open(filepath, 'rb') as f:
		G = pickle.load(f)

	# Read edge attribute data
	tpr_df = pd.read_csv(tpr_filepath)

	# Add edge attributes
	for _, row in tpr_df.iterrows():
		u, v = row['Up'], row['Down']
		if G.has_edge(u, v):
			G.edges[u, v].update({
				'mechanism': row['Name'],
				'number': row['Number']  # Ensure number attribute is correctly loaded
			})

	# Mark node types
	up_nodes = set(tpr_df['Up'])
	down_nodes = set(tpr_df['Down'])

	for node in G.nodes():
		if node in up_nodes and node in down_nodes:
			G.nodes[node]['type'] = 'both'
		elif node in up_nodes:
			G.nodes[node]['type'] = 'up'
		elif node in down_nodes:
			G.nodes[node]['type'] = 'down'
		else:
			G.nodes[node]['type'] = 'unknown'

	print(f"Graph loaded. Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
	return G


# ------------------------ Edge Width Normalization Function ------------------------
def normalize_edge_width(G, min_width=1.0, max_width=10.0):
	"""Normalize number values to specified width range"""
	# Correctly extract number values (including null value check)
	valid_edges = [(u, v, data) for u, v, data in G.edges(data=True) if 'number' in data]
	numbers = [data['number'] for u, v, data in valid_edges]

	# Return default width if no valid data
	if not numbers:
		return [min_width] * G.number_of_edges()

	# Calculate extremes
	min_num = min(numbers)
	max_num = max(numbers)

	# Handle single value case
	if max_num == min_num:
		return [max_width] * G.number_of_edges()

	# Generate width list (fix generator issue)
	widths = []
	for u, v, data in G.edges(data=True):
		if 'number' in data:
			n = data['number']
			width = ((n - min_num) / (max_num - min_num)) * (max_width - min_width) + min_width
		else:
			width = min_width
		widths.append(float(width))

	return widths

# ------------------------ Step 2: Initialize relation_type ------------------------
def initialize_edge_relations(G):
	for u, v in G.edges():
		G.edges[u, v]['relation_type'] = 'sequential'  # Initial default
	return G


# ------------------------ Step 3: Identify Sequential Relations A->B->C ------------------------
def identify_sequential_relations(G):
	sequential_relations = []
	for node in G.nodes():
		predecessors = list(G.predecessors(node))
		successors = list(G.successors(node))
		if len(predecessors) == 1 and len(successors) == 1:
			a = predecessors[0]
			b = node
			c = successors[0]
			# Record relationship + mechanism
			mech1 = G.edges[a, b].get('mechanism', 'Unknown')
			mech2 = G.edges[b, c].get('mechanism', 'Unknown')
			sequential_relations.append((a, b, c, mech1, mech2))
	print(f"Identified {len(sequential_relations)} sequential relations.")
	return sequential_relations


# ------------------------ Step 4: Identify Paternal Relations A->B, A->C ------------------------
def identify_paternal_relations(G):
	paternal_relations = []
	for node in G.nodes():
		successors = list(G.successors(node))
		if len(successors) >= 2:
			for i in range(len(successors)):
				for j in range(i + 1, len(successors)):
					b = successors[i]
					c = successors[j]
					mech1 = G.edges[node, b].get('mechanism', 'Unknown')
					mech2 = G.edges[node, c].get('mechanism', 'Unknown')
					paternal_relations.append((node, b, c, mech1, mech2))
	print(f"Identified {len(paternal_relations)} paternal relations.")
	return paternal_relations


# ------------------------ Step 5: Identify V-Shape Relations B->A, C->A ------------------------
def identify_v_shape_relations(G):
	v_shape_relations = []
	for node in G.nodes():
		predecessors = list(G.predecessors(node))
		if len(predecessors) >= 2:
			for i in range(len(predecessors)):
				for j in range(i + 1, len(predecessors)):
					b = predecessors[i]
					c = predecessors[j]
					mech1 = G.edges[b, node].get('mechanism', 'Unknown')
					mech2 = G.edges[c, node].get('mechanism', 'Unknown')
					v_shape_relations.append((b, node, c, mech1, mech2))
	print(f"Identified {len(v_shape_relations)} v-shape relations.")
	return v_shape_relations


# ------------------------ Step 6: Update Edge relation_type Labels ------------------------
def update_edge_relations(G, sequential_relations, v_shape_relations, paternal_relations):
	for a, b, c, _, _ in sequential_relations:
		if G.has_edge(a, b):
			G.edges[a, b]['relation_type'] = 'sequential'
		if G.has_edge(b, c):
			G.edges[b, c]['relation_type'] = 'sequential'
	for a, b, c, _, _ in paternal_relations:
		if G.has_edge(a, b):
			G.edges[a, b]['relation_type'] = 'paternal'
		if G.has_edge(a, c):
			G.edges[a, c]['relation_type'] = 'paternal'
	for b, a, c, _, _ in v_shape_relations:
		if G.has_edge(b, a):
			G.edges[b, a]['relation_type'] = 'v_shape'
		if G.has_edge(c, a):
			G.edges[c, a]['relation_type'] = 'v_shape'
	print("Edge relation types updated successfully.")
	return G


# ------------------------ Step 7: Save Graph ------------------------
def save_graph(G, filepath='PAEs_network_with_relations.gpickle'):
	with open(filepath, 'wb') as f:
		pickle.dump(G, f)
	print(f"Graph with relation types saved as '{filepath}'.")


# ------------------------ Step 8: Statistics by Mechanism ------------------------
def statistics_by_mechanism(sequential_relations, v_shape_relations, paternal_relations):
	stats = {}

	# Sequential relations
	for a, b, c, mech1, mech2 in sequential_relations:
		for mech in [mech1, mech2]:
			stats.setdefault(mech, {'Sequential': 0, 'V-Shape': 0, 'Paternal': 0})
			stats[mech]['Sequential'] += 1

	# V-shape relations
	for a, b, c, mech1, mech2 in v_shape_relations:
		for mech in [mech1, mech2]:
			stats.setdefault(mech, {'Sequential': 0, 'V-Shape': 0, 'Paternal': 0})
			stats[mech]['V-Shape'] += 1

	# Paternal relations
	for b, a, c, mech1, mech2 in paternal_relations:
		for mech in [mech1, mech2]:
			stats.setdefault(mech, {'Sequential': 0, 'V-Shape': 0, 'Paternal': 0})
			stats[mech]['Paternal'] += 1

	# Convert to DataFrame
	stats_df = pd.DataFrame.from_dict(stats, orient='index')
	stats_df.index.name = 'Mechanism'
	stats_df.reset_index(inplace=True)
	stats_df.to_csv('relation_statistics_by_mechanism.csv', index=False)
	print("Saved detailed mechanism-based statistics to 'relation_statistics_by_mechanism.csv'.")


# ------------------------ Export Node Data ------------------------
def export_nodes_data(G, filepath='nodes_info.csv'):
	"""Export node basic information"""
	nodes_data = []
	for node in G.nodes():
		nodes_data.append({
			'Node': node,
			'BaseProbability': 0.5  # Default base probability, can be adjusted as needed
		})
	nodes_df = pd.DataFrame(nodes_data)
	nodes_df.to_csv(filepath, index=False)
	print(f"Exported node data to {filepath}")


# ------------------------ Export Edge Data ------------------------
def export_edges_data(G, filepath='edges_info.csv'):
	"""Export edge relationships and calculate co-occurrence probability"""
	edges_data = []
	for u, v in G.edges():
		edges_data.append({
			'Source': u,
			'Target': v,
			'Mechanism': G.edges[u, v].get('mechanism', 'Unknown'),
			'CooccurrenceFrequency': G.edges[u, v].get('number', 0),
			'RelationType': G.edges[u, v]['relation_type']
		})
	edges_df = pd.DataFrame(edges_data)

	# Calculate normalized co-occurrence probability (based on all edge frequencies)
	total_freq = edges_df['CooccurrenceFrequency'].sum()
	edges_df['CooccurrenceProbability'] = edges_df['CooccurrenceFrequency'] / total_freq

	edges_df.to_csv(filepath, index=False)
	print(f"Exported edge data to {filepath}")


# ------------------------ Draw and Save Topology Graph ------------------------
def draw_topology_graph(G, output_file='topology_graph.png'):
	# Initialize canvas
	rcParams['figure.dpi'] = 300
	plt.figure(figsize=(30, 28))

	# Layout algorithm
	pos = nx.spring_layout(G, k=0.5, seed=42, iterations=50)

	# Node color mapping
	node_colors = []
	color_mapping = {
		'up': '#5CC9F5',
		'down': '#f89ccf',
		'both': '#9400D3',
		'unknown': 'gray'
	}

	# Edge settings
	edge_widths = normalize_edge_width(G)  # Get normalized edge widths
	edge_colors = []
	edge_color_map = {
		'sequential': '#257d8b',
		'paternal': '#ff7f0e',
		'v_shape': '#B8DBB3',
		'unknown': '#7f7f7f'
	}

	# Collect edge data
	for u, v in G.edges():
		edge_type = G.edges[u, v].get('relation_type', 'unknown')
		edge_colors.append(edge_color_map.get(edge_type, '#7f7f7f'))

	# Draw nodes
	nx.draw_networkx_nodes(
		G, pos,
		node_size=1800,
		node_color=[color_mapping[n[1]['type']] for n in G.nodes(data=True)],
		alpha=0.8,
		linewidths=0,
		#edgecolors='black'
	)

	# Draw edges (added width parameter)
	nx.draw_networkx_edges(
		G, pos,
		edge_color=edge_colors,
		width=edge_widths,  # Use normalized widths
		alpha=0.7,
		arrowstyle="-|>,head_length=0.5,head_width=0.2",
		arrowsize=15,
		connectionstyle='arc3,rad=0.1',
		node_size=1800,
		min_source_margin=5,
		min_target_margin=5
	)

	# Draw labels
	nx.draw_networkx_labels(
		G, pos,
		font_size=13,
		font_family='Arial',
		#font_weight='bold',
		#verticalalignment='bottom'
	)

	# Legend design (added edge width description)
	from matplotlib.lines import Line2D
	legend_elements = [
		Line2D([0], [0], marker='o', color='w', label='PAEs',
		       markerfacecolor='#5CC9F5', markersize=30),
		Line2D([0], [0], marker='o', color='w', label='mPAEs',
		       markerfacecolor='#f89ccf', markersize=30),
		#Line2D([0], [0], marker='o', color='w', label='Both Types',
		       #markerfacecolor='#9400D3', markersize=12),
		Line2D([0], [0], color='#257d8b', lw=5, label='Sequential'),
		Line2D([0], [0], color='#ef8b67', lw=5, label='Paternal'),
		Line2D([0], [0], color='#B8DBB3', lw=5, label='V-Shape'),
		#Line2D([0], [0], color='black', lw=4, label='Max Frequency'),
		#Line2D([0], [0], color='black', lw=1, label='Min Frequency')
	]

	plt.legend(handles=legend_elements, loc='upper right', fontsize=30,frameon=False) #title="Legend\n"
	plt.axis('off')
	plt.tight_layout()
	plt.savefig(output_file, bbox_inches='tight', dpi=300)
	plt.close()
	print(f"Topology graph saved as {output_file}")


# ------------------------ Main Function ------------------------
def main():
	G = load_graph()
	G = initialize_edge_relations(G)
	sequential_relations = identify_sequential_relations(G)
	v_shape_relations = identify_v_shape_relations(G)
	paternal_relations = identify_paternal_relations(G)
	G = update_edge_relations(G, sequential_relations, v_shape_relations, paternal_relations)
	save_graph(G)

	# New data export functions
	export_nodes_data(G)
	export_edges_data(G)
	statistics_by_mechanism(sequential_relations, v_shape_relations, paternal_relations)

	# New plotting function
	draw_topology_graph(G)


if __name__ == "__main__":
	main()