import pandas as pd
from collections import Counter
import numpy as np

def compute_y_true_metrics(
    gdf_sjoined: pd.DataFrame,
    subG,
    *,
    cell_col: str = "cell",
    time_col: str = "time_group",
    lag_hours: int = 12,
):
    """input: gdf_sjoined, subG
    output: 
        'y_true'        : {node: p_all},
        'y_true_sf'     : {node: p_sf},
        'y_true_network': {node: p_net},
        'y_true_sis'    : {node: p_sis}
    } 

    """
    
    node_set = set(subG.nodes())
    df = gdf_sjoined[[cell_col, time_col]].copy()
    df = df[df[cell_col].isin(node_set)]

   
    time_cnt = df[time_col].nunique()
   
    burning_by_time = (
        df.groupby(time_col)[cell_col]
          .apply(set)
          .to_dict()
    )

    lag = pd.Timedelta(hours=lag_hours)
    network_events = []
    sf_events = []

    for t, cells in burning_by_time.items():
        prev_cells = burning_by_time.get(t - lag, set())
        for node in cells:
            if prev_cells & set(subG.neighbors(node)):
                network_events.append((node, t))
            else:
                sf_events.append((node, t))

    
    def _counts(events):
        c = Counter([n for n, _ in events])
        return {n: c.get(n, 0) / time_cnt for n in node_set}

    y_true_network_dict = _counts(network_events)
    y_true_sf_dict      = _counts(sf_events)

    all_counts = (
        df.drop_duplicates()
          .groupby(cell_col)
          .size()
          .to_dict()
    )
    y_true_dict = {n: all_counts.get(n, 0) / time_cnt for n in node_set}


    y_true_sis_dict = {}
    for n in node_set:
        p, p_sf = y_true_dict[n], y_true_sf_dict[n]
        denom = 1 - p_sf
        y_true_sis_dict[n] = (p - p_sf) / denom if denom > 0 else 0.0

    return {
        'y_true_dict': y_true_dict,
        'y_true_sf_dict': y_true_sf_dict,
        'y_true_network_dict': y_true_network_dict,
        'y_true_sis_dict': y_true_sis_dict,
    }


