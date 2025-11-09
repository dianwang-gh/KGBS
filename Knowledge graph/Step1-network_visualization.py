import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pickle


# ------------------------- New Fix Point -------------------------
def point_to_segment_distance(x0, y0, x1, y1, x2, y2):
    dx, dy = x2 - x1, y2 - y1
    segment_length_sq = dx ** 2 + dy ** 2
    if segment_length_sq == 0:
        return ((x0 - x1) ** 2 + (y0 - y1) ** 2) ** 0.5
    t = ((x0 - x1) * dx + (y0 - y1) * dy) / segment_length_sq
    t = max(0, min(1, t))
    proj_x = x1 + t * dx
    proj_y = y1 + t * dy
    return ((x0 - proj_x) ** 2 + (y0 - proj_y) ** 2) ** 0.5


# ------------------------------------------------------------

# Step 1: Load Data from CSV files
def load_data():
    PAEs_df = pd.read_csv('PAEs_TP.csv')
    TPR_df = pd.read_csv('TPR_number.csv')
    return PAEs_df, TPR_df


# Step 2: Build Network using NetworkX
def build_network(PAEs_df, TPR_df, name_col=None):
    G = nx.DiGraph()

    # Add nodes
    for _, row in TPR_df.iterrows():
        G.add_node(row['Up'], type='PAEs')
        G.add_node(row['Down'], type='mPAEs')

    # Add PAE-mPAE transformation paths
    for _, row in PAEs_df.iterrows():
        for i in range(1, 10):
            upstream_col = 'PAE'
            downstream_col = f'TP{i}'
            name_col = 'Name'
            if upstream_col in row and downstream_col in row:
                upstream_event = row[upstream_col]
                downstream_event = row[downstream_col]
                name_event = row[name_col]
                if pd.notna(upstream_event) and pd.notna(downstream_event):
                    u_type = G.nodes.get(upstream_event, {}).get('type', 'Unknown')
                    v_type = G.nodes.get(downstream_event, {}).get('type', 'Unknown')
                    if u_type == 'PAEs' and v_type == 'mPAEs':
                        G.add_edge(
                            upstream_event, downstream_event,
                            relation=name_event,
                            relation_type='sequential'  # Default added as sequential relationship
                        )
    return G


def visualize_network(G, save_path='PAEs_network_visualization.png'):
    tpr_df = pd.read_csv('TPR_modified_number.csv')
    name_mapping = {(row['Up'], row['Down']): row['Name'] for _, row in tpr_df.iterrows()}
    number_mapping = {(row['Up'], row['Down']): row['Number'] for _, row in tpr_df.iterrows()}

    # Dynamically calculate edge width
    max_number = tpr_df['Number'].max()
    min_number = tpr_df['Number'].min()
    edge_widths = [
        (number_mapping.get((u, v), min_number) - min_number) / (max_number - min_number) * 9 + 1
        for u, v in G.edges()
    ] if max_number != min_number else [5] * len(G.edges())

    # Define edge color mapping
    relation_colors = {
        'sequential': 'gray',
        'v_shape': 'green',
        'paternal': 'red'
    }
    edge_colors = [
        relation_colors.get(G.edges[u, v].get('relation_type', 'sequential'), 'gray')
        for u, v in G.edges()
    ]

    plt.switch_backend('TkAgg')
    pos = nx.spring_layout(G, k=0.01, iterations=50)
    fig, ax = plt.subplots(figsize=(20, 20))

    # Draw nodes
    PAEs_nodes = [node for node, attr in G.nodes(data=True) if attr.get('type') == 'PAEs']
    mPAEs_nodes = [node for node, attr in G.nodes(data=True) if attr.get('type') == 'mPAEs']
    nx.draw_networkx_nodes(G, pos, nodelist=PAEs_nodes, node_size=300, node_color='#5CC9F5', alpha=0.8, ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=mPAEs_nodes, node_size=300, node_color='#F78AE0', alpha=0.8, ax=ax)

    # Draw edges (set colors according to different relation_types)
    nx.draw_networkx_edges(
        G, pos,
        edgelist=G.edges(),
        width=edge_widths,
        edge_color=edge_colors,
        alpha=0.8,
        ax=ax,
        arrows=True,
        arrowstyle='-|>',
        arrowsize=15
    )

    nx.draw_networkx_labels(G, pos, font_size=6, font_color='black', ax=ax)

    # Legend
    blue_line = plt.Line2D([], [], marker='o', color='w', markerfacecolor='#5CC9F5', markersize=10, label='PAEs')
    pink_line = plt.Line2D([], [], marker='o', color='w', markerfacecolor='#F78AE0', markersize=10, label='mPAEs')
    gray_line = plt.Line2D([], [], color='gray', lw=2, label='Sequential')
    green_line = plt.Line2D([], [], color='green', lw=2, label='V-Shape')
    red_line = plt.Line2D([], [], color='red', lw=2, label='Paternal')
    plt.legend(handles=[blue_line, pink_line, gray_line, green_line, red_line], frameon=False, labelspacing=1,
               title='Legend')

    # Click event
    def on_click(event):
        if event.inaxes is not None:
            edge_threshold = 0.05
            min_edge_distance = float('inf')
            clicked_edge = None
            for u, v in G.edges():
                x1, y1 = pos[u]
                x2, y2 = pos[v]
                distance = point_to_segment_distance(event.xdata, event.ydata, x1, y1, x2, y2)
                if distance < edge_threshold and distance < min_edge_distance:
                    min_edge_distance = distance
                    clicked_edge = (u, v)

            if clicked_edge:
                relation = name_mapping.get(clicked_edge, 'N/A')
                number = number_mapping.get(clicked_edge, 'N/A')
                relation_type = G.edges[clicked_edge].get('relation_type', 'sequential')
                print(
                    f"Edge ({clicked_edge[0]} → {clicked_edge[1]}) clicked! Relation: {relation}, Number: {number}, Type: {relation_type}")
                return

            clicked_node = None
            min_node_distance = float('inf')
            for node, (x, y) in pos.items():
                node_size = 300
                threshold = (node_size / 2000) ** 2
                distance = (event.xdata - x) ** 2 + (event.ydata - y) ** 2
                if distance < threshold and distance < min_node_distance:
                    min_node_distance = distance
                    clicked_node = node

            if clicked_node:
                print(f"Node ({clicked_node}) clicked! Type: {G.nodes[clicked_node].get('type', 'Unknown')}")
                return

    fig.canvas.mpl_connect('button_press_event', on_click)
    plt.axis('off')
    plt.savefig(save_path, format="PNG", dpi=300, bbox_inches="tight")
    plt.show()


def main():
    PAEs_df, TPR_df = load_data()
    G = build_network(PAEs_df, TPR_df)
    visualize_network(G)

    # Save graph
    with open('PAEs_network.gpickle', 'wb') as f:
        pickle.dump(G, f)

    print("Graph has been saved to 'PAEs_network.gpickle'.")

if __name__ == "__main__":
    main()
