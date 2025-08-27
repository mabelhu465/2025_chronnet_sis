# Assess Fire Risk Using Chronnet and SIS


## Features

- **Network Construction & Pruning:** Build spatio‑temporal networks (chronnets) from geospatial fire archives using [chronnet_utils.py](chronnet_utils.py).
- **Fire Event Analysis:** Compute network-driven, spontaneous fire events.
- **Node-Level Metrics:** Calculate centrality measures (e.g. degree, PageRank, eigenvector, betweenness, closeness) and clustering coefficients for network nodes.
- **SIS Modeling:** Compute the steady‑state infection rate in networks via the SIS model ([sis_steady_state_and_eval.py](sis_steady_state_and_eval.py)).
- **Jupyter Notebooks:** Interactive notebooks, including [main_experiment.ipynb](main_experiment.ipynb) and [ablation_experiment_without_selfloop.ipynb](ablation_experiment_without_selfloop.ipynb), demonstrate the full experimental workflow.

## Installation

1. **Clone the repository:**

   ```sh
   git clone https://github.com/yourusername/chronnet-fire-risk-modeling.git
   cd chronnet-fire-risk-modeling
   ```

2. **Create a virtual environment (optional but recommended):**

   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. **Install dependencies:**

   The project uses several Python libraries including GeoPandas, NetworkX, Pandas, NumPy, Matplotlib, Seaborn, SciPy, and scikit-learn. Install all required packages from [requirements.txt](requirements.txt):

   ```sh
   pip install -r requirements.txt
   ```

## Configuration

- **Data Sources:**
  - **Shapefile:** Update the path to your fire archive shapefile in the notebooks (e.g., in [main_experiment.ipynb](main_experiment.ipynb), search for the `shapefile_path` variable).
  - **Fire Data Filters:** Adjust filters (e.g., `CONFIDENCE` field) as needed to suit your dataset.


## Usage

- Open and run the notebooks in Jupyter or an IDE that supports interactive notebooks (e.g., Visual Studio Code with the Python extension). Start with [main_experiment.ipynb](main_experiment.ipynb) to follow the end-to-end workflow.

- After processing, review the generated CSV files (e.g., `sis_results.csv`, `scc_eval_results.csv`, `best_tau_by_grid_scc.csv`) for detailed results and visualizations.

## Repository Structure

- **main_experiment.ipynb:** Primary notebook demonstrating the end-to-end analysis.
- **ablation_experiment_without_selfloop.ipynb:** Notebook for ablation studies.
- **chronnet_utils.py:** Utility functions for creating hexagonal grids, building chronnets, and pruning networks.
- **compute_y_true.py:** Module for computing ground truth fire metrics.
- **sis_steady_state_and_eval.py:** Implementation of the SIS steady‑state model and evaluation metrics.
- **requirements.txt:** List of project dependencies.
- **\_\_pycache\_\_/**: Directory containing cached bytecode files.

