
import geopandas as gpd
import pandas as pd
import networkx as nx
import numpy as np
from shapely.geometry import Polygon





def create_hex_grid(gdf, hex_size=5000):
    """
    Generate a flat-topped hexagonal grid covering the bounding box of the input GeoDataFrame.
    The hexagons are arranged edge-to-edge.
    
    Parameters:
      gdf: GeoDataFrame with spatial data.
      hex_size: The side length of the hexagon (also the distance from the center to a vertex) in meters.
      
    Returns:
      A GeoDataFrame of hexagons with a unique 'cell' ID.
    """
    # Get the bounding box of the input data: [minx, miny, maxx, maxy]
    minx, miny, maxx, maxy = gdf.total_bounds
    
    # Flat-topped hexagon spacing
    dx = 1.5 * hex_size           # Horizontal spacing
    dy = np.sqrt(3) * hex_size    # Vertical spacing

    hexagons = []
    centers = []

    # Generate x coordinates for grid centers
    x_coords = np.arange(minx, maxx + dx, dx)

    # Adjust y coordinates to avoid missing hexagons at the boundary
    y_coords = np.arange(miny - dy / 2, maxy + dy, dy)

    # Generate grid of hexagon centers
    for i, x in enumerate(x_coords):
        offset = dy / 2 if i % 2 == 1 else 0
        for y in y_coords:
            centers.append((x, y + offset))

    # Generate flat-topped hexagons
    for (x_center, y_center) in centers:
        angles = np.linspace(0, 2 * np.pi, 7)  # No need to rotate manually
        hexagon = Polygon([
            (x_center + hex_size * np.cos(a), y_center + hex_size * np.sin(a))
            for a in angles
        ])
        hexagons.append(hexagon)

    hex_grid = gpd.GeoDataFrame({'geometry': hexagons}, crs=gdf.crs)
    hex_grid['cell'] = hex_grid.index.astype(str)
    return hex_grid







def build_chronnet_freq(df,dmax=20000,freq='12h'):
    """
    Build a Chronnet directed graph from fire event data.
    
    This function connects consecutive events in time (h = 1) only if the distance 
    between two consecutive events is less than or equal to dmax.
    
    Parameters:
      df: GeoDataFrame containing fire events with 'acq_time', 'cell', and 'geometry' columns.
      dmax: Maximum allowed distance (in meters) between consecutive events to create a link (default: 20000 m, i.e., 20 km).
      freq: Time frequency for grouping events (default: '12h').
      
    Returns:
      A directed NetworkX graph where nodes represent hexagon cells and edges represent transitions 
      between consecutive events that are within the specified distance threshold.
    """
    G = nx.DiGraph()
    
    df=df.copy()
    df['centroid'] = df['geometry'].apply(lambda geom: geom.centroid)
    df['time_group'] = df['acq_time'].dt.floor(freq)
    df_sorted = df.sort_values(by='time_group').reset_index(drop=True)

    unique_times = np.sort(df_sorted['time_group'].unique())
    n= len(unique_times)

    # For h = 1, connect each event to its immediate successor.
    for i in range(n-1):
        t_i = unique_times[i]
        t_j = unique_times[i + 1]

        # Skip if the time difference is greater than 100 days
        days = (t_j - t_i) / np.timedelta64(1, 'D')
        if days > 100:
            continue


        df_i = df_sorted[df_sorted['time_group'] == t_i].reset_index(drop=True)
        df_j = df_sorted[df_sorted['time_group'] == t_j].reset_index(drop=True)

        centers_i = np.vstack(df_i['centroid'].apply(lambda geom: [geom.x, geom.y]).values)
        centers_j = np.vstack(df_j['centroid'].apply(lambda geom: [geom.x, geom.y]).values)

        dx = centers_i[:, 0][:,None] - centers_j[:, 0][None,:]
        dy = centers_i[:, 1][:,None] - centers_j[:, 1][None,:]
        dist_matrix = np.sqrt(dx**2 + dy**2)

        valid_i, valid_j = np.where(dist_matrix <= dmax)
        for idx in range(len(valid_i)):
            source = df_i.loc[valid_i[idx],'cell']
            target = df_j.loc[valid_j[idx],'cell']

            if source not in G:
                G.add_node(source, lon=centers_i[valid_i[idx], 0], lat=centers_i[valid_i[idx], 1])
            if target not in G:
                G.add_node(target, lon=centers_j[valid_j[idx], 0], lat=centers_j[valid_j[idx], 1])
    
            if G.has_edge(source, target):
                G[source][target]['weight'] += 1
            else:
                G.add_edge(source, target, weight=1)
    
            
 
    
    return G



def build_chronnet(df,dmax=20000):
    """
    Build a Chronnet directed graph from fire event data.
    
    This function connects consecutive events in time (h = 1) only if the distance 
    between two consecutive events is less than or equal to dmax.
    
    Parameters:
      df: GeoDataFrame containing fire events with 'acq_time', 'cell', and 'geometry' columns.
      dmax: Maximum allowed distance (in meters) between consecutive events to create a link (default: 10000 m, i.e., 10 km).
      
    Returns:
      A directed NetworkX graph where nodes represent hexagon cells and edges represent transitions 
      between consecutive events that are within the specified distance threshold.
    """
    G = nx.DiGraph()
    
    df=df.copy()
    df['centroid'] = df['geometry'].apply(lambda geom: geom.centroid)
    
    df_sorted = df.sort_values(by='acq_time').reset_index(drop=True)

    unique_times = np.sort(df_sorted['acq_time'].unique())
    n= len(unique_times)

    # For h = 1, connect each event to its immediate successor.
    for i in range(n-1):
        t_i = unique_times[i]
        t_j = unique_times[i + 1]

        # Skip if the time difference is greater than 100 days
        days = (t_j - t_i) / np.timedelta64(1, 'D')
        if days > 100:
            continue

        df_i = df_sorted[df_sorted['acq_time'] == t_i].reset_index(drop=True)
        df_j = df_sorted[df_sorted['acq_time'] == t_j].reset_index(drop=True)

        centers_i = np.vstack(df_i['centroid'].apply(lambda geom: [geom.x, geom.y]).values)
        centers_j = np.vstack(df_j['centroid'].apply(lambda geom: [geom.x, geom.y]).values)

        dx = centers_i[:, 0][:,None] - centers_j[:, 0][None,:]
        dy = centers_i[:, 1][:,None] - centers_j[:, 1][None,:]
        dist_matrix = np.sqrt(dx**2 + dy**2)

        valid_i, valid_j = np.where(dist_matrix <= dmax)
        for idx in range(len(valid_i)):
            source = df_i.loc[valid_i[idx],'cell']
            target = df_j.loc[valid_j[idx],'cell']

            if source not in G:
                G.add_node(source, lon=centers_i[valid_i[idx], 0], lat=centers_i[valid_i[idx], 1])
            if target not in G:
                G.add_node(target, lon=centers_j[valid_j[idx], 0], lat=centers_j[valid_j[idx], 1])
    
            if G.has_edge(source, target):
                G[source][target]['weight'] += 1
            else:
                G.add_edge(source, target, weight=1)
    
            
 
    
    return G




def prune_chronnet(G, min_weight=2):
    """
    Remove edges with weight below a threshold.
    
    Parameters:
      G: A NetworkX DiGraph.
      min_weight: Minimum edge weight to retain.
      
    Returns:
      A pruned directed graph.
    """
    G_pruned = nx.DiGraph()
    for u, v, d in G.edges(data=True):
        if d['weight'] >= min_weight:
            G_pruned.add_edge(u, v, weight=d['weight'])
    return G_pruned

