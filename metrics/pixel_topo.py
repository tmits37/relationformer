import cv2
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# Load segmented image (binary mask)
segmented_image = cv2.imread('segmented_image.png', 0)  # 0 for grayscale

# Find contours
contours, _ = cv2.findContours(segmented_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Initialize an empty graph
G = nx.Graph()

# Iterate over contours and add edges
for i, contour in enumerate(contours):
    # Approximate contour to simplify the shape
    epsilon = 0.01 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)

    # Add nodes and edges to the graph
    for j in range(len(approx)):
        # Add node
        node = tuple(approx[j][0])
        G.add_node(node)

        # Connect nodes with an edge
        if j < len(approx) - 1:
            next_node = tuple(approx[j + 1][0])
            G.add_edge(node, next_node)

# Optional: visualize the graph
pos = {node: (node[0], -node[1]) for node in G.nodes()}  # Flip y-axis for display
nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=500)
plt.show()

# Number of connected components
num_components = nx.number_connected_components(G)
print("Number of connected components:", num_components)

# Analyze each component (optional)
for component in nx.connected_components(G):
    subgraph = G.subgraph(component)
    # Further analysis like checking cycles, length, etc.
    
def calculate_topological_f1(predicted_graph, ground_truth_graph, match_criteria):
    tp = 0  # True Positives
    fp = 0  # False Positives
    fn = 0  # False Negatives

    # Assuming match_criteria is a function to determine if a predicted element matches with ground truth
    for predicted_element in predicted_graph:
        if match_criteria(predicted_element, ground_truth_graph):
            tp += 1
        else:
            fp += 1

    for ground_truth_element in ground_truth_graph:
        if not match_criteria(ground_truth_element, predicted_graph):
            fn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1_score