import numpy as np
from shapely.geometry import LineString
import pandas as pd


def distance_from_ref(point):
    return np.sqrt(point[0]**2 + point[1]**2)


def reorient_linestring(line):
    start, end = line.coords[0], line.coords[-1]
    if distance_from_ref(start) > distance_from_ref(end):
        # Reverse the LineString
        return LineString([end, start])
    return line


def generate_directed_graph_and_sorting(gdf):
    gdf = gdf.copy()
    gdf['geometry'] = gdf['geometry'].apply(reorient_linestring)
    gdf['centroid_x'] = gdf['geometry'].centroid.x
    gdf['centroid_y'] = gdf['geometry'].centroid.y
    unique_gdf = gdf.groupby(['centroid_x',
                              'centroid_y']).first().reset_index(drop=True)
    return unique_gdf


def gdf_to_nodes_and_edges_linestring(gdf):
    nodes = []
    for _, row in gdf.iterrows():
        polygon = row['geometry']
        if polygon.geom_type == 'LineString':
            for x, y in polygon.coords:
                nodes.append((x, y))
        elif polygon.geom_type == 'MultiLineString':
            for part in polygon:
                for x, y in part.coords:
                    nodes.append((x, y))
        else:
            raise AttributeError

    # Remove duplicates if necessary
    nodes = list(set(nodes))

    # Create a DataFrame for nodes with unique indices
    node_df = pd.DataFrame(nodes, columns=['x', 'y'])
    node_df['node_id'] = range(len(node_df))

    edges = []
    for _, row in gdf.iterrows():
        polygon = row['geometry']
        if polygon.geom_type == 'LineString':
            coords = polygon.coords  # Exclude closing vertex
            edge = [
                (node_df[(node_df['x'] == x) & (node_df['y'] == y)].index[0],
                 node_df[(node_df['x'] == coords[(i + 1) % len(coords)][0])
                         & (node_df['y'] == coords[(i + 1) %
                                                   len(coords)][1])].index[0])
                for i, (x, y) in enumerate(coords)
            ]
            # original code is assumed "closed" polygon
            edge = edge[:-1]
            edges.extend(edge)
        elif polygon.geom_type == 'MultiLineString':
            for part in polygon:
                coords = part.coords
                edge = [
                    (node_df[(node_df['x'] == x)
                             & (node_df['y'] == y)].index[0],
                     node_df[(node_df['x'] == coords[(i + 1) % len(coords)][0])
                             & (node_df['y'] == coords[
                                 (i + 1) % len(coords)][1])].index[0])
                    for i, (x, y) in enumerate(coords)
                ]
                # original code is assumed "closed" polygon
                edge = edge[:-1]
                edges.extend(edge)

    return node_df[['y', 'x']].values, edges