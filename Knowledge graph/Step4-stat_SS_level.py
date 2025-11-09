import networkx as nx
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.patches import Patch

# ------------------------ Color Configuration ------------------------
SCREENING_COLORS = {
	'Level 1': '#b52b36',  # Dark red
	'Level 2': '#DC143C',  # Bright red
	'Level 3': '#FF6347',  # Tomato red
	'Level 4': '#FFA07A',  # Light coral red
	'ND': '#e2d7d7'  # Gray
}

TYPE_BORDER_COLORS = {
	'up': '#5CC9F5',  # Dodger blue (outline)
	'down': '#f89ccf',  # Hot pink (outline)
	'unknown': 'gray'  # Default
}


# ------------------------ Modified Loading Function ------------------------
def load_graph(filepath='PAEs_network.gpickle',
               tpr_filepath='TPR_number.csv',
               node_info_path='nodes_level.csv'):
	# Load graph structure
	with open(filepath, 'rb') as f:
		G = pickle.load(f)

	# ================== Edge Attribute Processing ==================
	tpr_df = pd.read_csv(tpr_filepath)
	for _, row in tpr_df.iterrows():
		u, v = row['Up'], row['Down']
		if G.has_edge(u, v):
			G.edges[u, v].update({
				'mechanism': row['Name'],
				'number': row['Number']
			})

	# ================== Node Attribute Processing ==================
	try:
		node_df = pd.read_csv(node_info_path)

		# Column name validation
		required_cols = ['Node', 'Type', 'ScreeningResult']
		if not all(col in node_df.columns for col in required_cols):
			missing = [col for col in required_cols if col not in node_df.columns]
			raise ValueError(f"CSV file missing required columns: {missing}")

		# Build node attribute dictionary
		node_dict = node_df.set_index('Node').to_dict(orient='index')

		# Update node attributes
		for node in G.nodes():
			if node in node_dict:
				data = node_dict[node]
				# Update Type (outline color)
				node_type = str(data.get('Type', 'unknown')).strip().lower()
				G.nodes[node]['type'] = node_type if node_type in TYPE_BORDER_COLORS else 'unknown'

				# Update ScreeningResult (inner color)
				screening = str(data.get('ScreeningResult', 'ND')).strip()
				G.nodes[node]['screening'] = screening if screening in SCREENING_COLORS else 'ND'
			else:
				# Default value handling
				G.nodes[node]['type'] = 'unknown'
				G.nodes[node]['screening'] = 'ND'

	except FileNotFoundError:
		print(f"Error: Node info file {node_info_path} not found, using default values for all nodes")
		for node in G.nodes():
			G.nodes[node]['type'] = 'unknown'
			G.nodes[node]['screening'] = 'ND'
	except Exception as e:
		print(f"Error occurred while loading node data: {str(e)}, using default values")
		for node in G.nodes():
			G.nodes[node].setdefault('type', 'unknown')
			G.nodes[node].setdefault('screening', 'ND')

	# ================== Final Attribute Validation ==================
	valid_types = {'up', 'down', 'unknown'}
	valid_screening = set(SCREENING_COLORS.keys())

	for node in G.nodes():
		# Ensure type is valid
		current_type = G.nodes[node].get('type', 'unknown')
		if current_type not in valid_types:
			G.nodes[node]['type'] = 'unknown'

		# Ensure screening result is valid
		current_screening = G.nodes[node].get('screening', 'ND')
		if current_screening not in valid_screening:
			G.nodes[node]['screening'] = 'ND'

	print(f"Graph loading completed | Nodes: {G.number_of_nodes()} | Edges: {G.number_of_edges()}")
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

# ------------------------ Initialize relation_type ------------------------
def initialize_edge_relations(G):
	for u, v in G.edges():
		G.edges[u, v]['relation_type'] = 'sequential'  # Initial default
	return G


# ------------------------ Identify Sequential Relations A->B->C ------------------------
def identify_sequential_relations(G):
	sequential_relations = []
	for node in G.nodes():
		predecessors = list(G.predecessors(node))
		successors = list(G.successors(node))
		if len(predecessors) == 1 and len(successors) == 1:
			a = predecessors[0]
			b = node
			c = successors[0]
			# Record relationship and mechanism
			mech1 = G.edges[a, b].get('mechanism', 'Unknown')
			mech2 = G.edges[b, c].get('mechanism', 'Unknown')
			sequential_relations.append((a, b, c, mech1, mech2))
	print(f"Identified {len(sequential_relations)} sequential relations.")
	return sequential_relations


# ------------------------ Identify Paternal Relations A->B, A->C ------------------------
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


# ------------------------ Identify V-Shape Relations B->A, C->A ------------------------
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


# ------------------------ Update Edge relation_type Labels ------------------------
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


# ------------------------ Save Graph ------------------------
def save_graph(G, filepath='PAEs_network_with_relations.gpickle'):
	with open(filepath, 'wb') as f:
		pickle.dump(G, f)
	print(f"Graph with relation types saved as '{filepath}'.")


# ------------------------ Statistics by Mechanism ------------------------
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
def export_nodes_data(G, filepath='nodes_info_new_0527.csv'):
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
def export_edges_data(G, filepath='edges_info_new.csv'):
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
def draw_topology_graph(G, output_file='topology_graph_level.png'):
	rcParams['figure.dpi'] = 300
	plt.figure(figsize=(30, 28))

	# Layout algorithm
	pos = nx.spring_layout(G, k=0.5, seed=42, iterations=50)

	# Prepare color data
	node_colors = []
	border_colors = []
	for node in G.nodes(data=True):
		# Inner color (ScreeningResult)
		screening = node[1]['screening']
		node_colors.append(SCREENING_COLORS.get(screening, '#808080'))

		# Outline color (Type)
		node_type = node[1]['type']
		border_colors.append(TYPE_BORDER_COLORS.get(node_type, 'gray'))

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

	node_sizes = [1800] * len(G.nodes())
	# Draw nodes
	nx.draw_networkx_nodes(
		G, pos,
		node_size=node_sizes,
		node_color=node_colors,
		edgecolors=border_colors,
		linewidths=3,
		alpha=0.9
	)

	nx.draw_networkx_edges(
		G, pos,
		edge_color=edge_colors,
		node_size=node_sizes,
		width=edge_widths,
		alpha=0.7,
		arrowstyle="-|>,head_length=0.5,head_width=0.2",
		arrowsize=15,
		connectionstyle='arc3,rad=0.1',
		min_source_margin=15,
		min_target_margin=15
	)

	# Draw labels
	nx.draw_networkx_labels(
		G, pos,
		font_size=13,
		font_family='Arial',
		#font_weight='bold',
		#verticalalignment='center'
	)
	from matplotlib.lines import Line2D
	# Build legend
	legend_elements = [
		Patch(facecolor='white', edgecolor=TYPE_BORDER_COLORS['up'],
		      label='PAEs', linewidth=3),
		Patch(facecolor='white', edgecolor=TYPE_BORDER_COLORS['down'],
		      label='mPAEs', linewidth=3),
		Line2D([0], [0], color='#257d8b', lw=5, label='Sequential'),
		Line2D([0], [0], color='#ef8b67', lw=5, label='Paternal'),
		Line2D([0], [0], color='#B8DBB3', lw=5, label='V-Shape'),
		Line2D([], [], color='white', alpha=0.0, label=""), # Use empty Line2D as placeholder for blank line
		*[Patch(facecolor=color, label=level) for level, color in SCREENING_COLORS.items()]
	]

	plt.legend(
		handles=legend_elements,
		loc='upper right',
		ncol=2,
		fontsize=30,
		#title="Legend Description",
		#title_fontsize=16,
		framealpha=0.9,
		frameon = False
	)

	plt.axis('off')
	plt.tight_layout()
	plt.savefig(output_file, bbox_inches='tight', dpi=300)
	plt.close()
	print(f"Topology graph saved to {output_file}")


# ------------------------ Main Function ------------------------
def main():
	G = load_graph()
	G = initialize_edge_relations(G)
	sequential_relations = identify_sequential_relations(G)
	v_shape_relations = identify_v_shape_relations(G)
	paternal_relations = identify_paternal_relations(G)
	G = update_edge_relations(G, sequential_relations, v_shape_relations, paternal_relations)
	save_graph(G)

	export_nodes_data(G)
	export_edges_data(G)
	statistics_by_mechanism(sequential_relations, v_shape_relations, paternal_relations)

	draw_topology_graph(G)


if __name__ == "__main__":
	main()