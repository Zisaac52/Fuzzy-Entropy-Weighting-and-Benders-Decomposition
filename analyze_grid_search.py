import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import argparse
import os

def analyze_results(results_file, nodes, epochs):
    """
    Reads grid search results, analyzes them, and generates plots.

    Args:
        results_file (str): Path to the CSV file containing grid search results.
        nodes (int): Number of nodes used in the experiments (for filtering/verification).
        epochs (int): Number of epochs used in the experiments (for filtering/verification).
    """
    print(f"Analyzing results from: {results_file}")
    print(f"Expected parameters: Nodes={nodes}, Epochs={epochs}") # Will print Epochs=2 by default now

    if not os.path.exists(results_file):
        print(f"Error: Results file not found at {results_file}")
        return

    try:
        df = pd.read_csv(results_file)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return

    # --- Data Cleaning and Filtering ---
    print("Original data shape:", df.shape)

    # Filter based on expected parameters (optional but good practice)
    df_filtered = df[(df['Nodes'] == nodes) & (df['Epochs'] == epochs) & (df['Method'] == 'fuzzy')]
    print("Filtered data shape (Nodes, Epochs, Method):", df_filtered.shape)

    if df_filtered.empty:
        print(f"Error: No matching data found for Nodes={nodes} and Epochs={epochs}.") # Error message reflects expected epochs=2
        return

    # Convert Accuracy and Loss to numeric, coercing errors to NaN
    df_filtered['Accuracy'] = pd.to_numeric(df_filtered['Accuracy'], errors='coerce')
    df_filtered['Loss'] = pd.to_numeric(df_filtered['Loss'], errors='coerce')

    # Drop rows where conversion failed (e.g., 'EvalError', 'CopyError')
    initial_rows = len(df_filtered)
    df_filtered.dropna(subset=['Accuracy', 'Loss'], inplace=True)
    dropped_rows = initial_rows - len(df_filtered)
    if dropped_rows > 0:
        print(f"Dropped {dropped_rows} rows due to non-numeric Accuracy/Loss values.")

    if df_filtered.empty:
        print("Error: No valid numeric data remaining after cleaning.")
        return

    # Define the expected m and r values for pivoting
    m_values_expected = sorted(df_filtered['FuzzyM'].unique())
    r_values_expected = sorted(df_filtered['FuzzyR'].unique())
    print("Unique FuzzyM values found:", m_values_expected)
    print("Unique FuzzyR values found:", r_values_expected)


    # --- Pivot Data for Plotting ---
    try:
        accuracy_pivot = df_filtered.pivot_table(index='FuzzyM', columns='FuzzyR', values='Accuracy')
        loss_pivot = df_filtered.pivot_table(index='FuzzyM', columns='FuzzyR', values='Loss')

        # Reindex to ensure all expected m/r values are present, fill missing with NaN
        accuracy_pivot = accuracy_pivot.reindex(index=m_values_expected, columns=r_values_expected)
        loss_pivot = loss_pivot.reindex(index=m_values_expected, columns=r_values_expected)

    except Exception as e:
        print(f"Error pivoting data: {e}")
        print("Check if there are duplicate (m, r) entries or other data issues.")
        return

    print("\nAccuracy Pivot Table Head:\n", accuracy_pivot.head())
    print("\nLoss Pivot Table Head:\n", loss_pivot.head())

    # Create directory for plots
    plot_dir = f"plots_node{nodes}_epoch{epochs}" # Directory name will now reflect epoch=2
    os.makedirs(plot_dir, exist_ok=True)
    print(f"\nSaving plots to directory: {plot_dir}")

    # --- Generate Plots ---
    m_grid, r_grid = np.meshgrid(r_values_expected, m_values_expected) # Note: meshgrid expects x, y

    # 1. Accuracy Plots
    if not accuracy_pivot.isnull().all().all():
        acc_data = accuracy_pivot.values # Get numpy array

        plt.figure(figsize=(8, 6))
        sns.heatmap(accuracy_pivot, annot=True, fmt=".2f", cmap="viridis", cbar_kws={'label': 'Accuracy (%)'})
        plt.title(f'Accuracy Heatmap (Nodes={nodes}, Epochs={epochs})') # Title will now reflect epoch=2
        plt.xlabel('Fuzzy R')
        plt.ylabel('Fuzzy M')
        plt.savefig(os.path.join(plot_dir, "accuracy_heatmap.png"))
        plt.close()
        print("Generated accuracy_heatmap.png")

        plt.figure(figsize=(8, 6))
        contour = plt.contourf(r_grid, m_grid, acc_data, cmap="viridis", levels=15)
        plt.colorbar(contour, label='Accuracy (%)')
        plt.contour(r_grid, m_grid, acc_data, colors='black', levels=15, linewidths=0.5)
        plt.title(f'Accuracy Contour Plot (Nodes={nodes}, Epochs={epochs})') # Title will now reflect epoch=2
        plt.xlabel('Fuzzy R')
        plt.ylabel('Fuzzy M')
        plt.xticks(r_values_expected)
        plt.yticks(m_values_expected)
        plt.savefig(os.path.join(plot_dir, "accuracy_contour.png"))
        plt.close()
        print("Generated accuracy_contour.png")

        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(r_grid, m_grid, acc_data, cmap="viridis", edgecolor='none')
        fig.colorbar(surf, shrink=0.5, aspect=5, label='Accuracy (%)')
        ax.set_title(f'Accuracy 3D Surface Plot (Nodes={nodes}, Epochs={epochs})') # Title will now reflect epoch=2
        ax.set_xlabel('Fuzzy R')
        ax.set_ylabel('Fuzzy M')
        ax.set_zlabel('Accuracy (%)')
        ax.set_xticks(r_values_expected)
        ax.set_yticks(m_values_expected)
        plt.savefig(os.path.join(plot_dir, "accuracy_3d_surface.png"))
        plt.close()
        print("Generated accuracy_3d_surface.png")
    else:
        print("Skipping Accuracy plots due to missing data.")


    # 2. Loss Plots
    if not loss_pivot.isnull().all().all():
        loss_data = loss_pivot.values # Get numpy array

        plt.figure(figsize=(8, 6))
        # Use a reversed colormap for loss (lower is better) e.g., 'viridis_r'
        sns.heatmap(loss_pivot, annot=True, fmt=".4f", cmap="viridis_r", cbar_kws={'label': 'Average Loss'})
        plt.title(f'Loss Heatmap (Nodes={nodes}, Epochs={epochs})') # Title will now reflect epoch=2
        plt.xlabel('Fuzzy R')
        plt.ylabel('Fuzzy M')
        plt.savefig(os.path.join(plot_dir, "loss_heatmap.png"))
        plt.close()
        print("Generated loss_heatmap.png")

        plt.figure(figsize=(8, 6))
        contour = plt.contourf(r_grid, m_grid, loss_data, cmap="viridis_r", levels=15)
        plt.colorbar(contour, label='Average Loss')
        plt.contour(r_grid, m_grid, loss_data, colors='black', levels=15, linewidths=0.5)
        plt.title(f'Loss Contour Plot (Nodes={nodes}, Epochs={epochs})') # Title will now reflect epoch=2
        plt.xlabel('Fuzzy R')
        plt.ylabel('Fuzzy M')
        plt.xticks(r_values_expected)
        plt.yticks(m_values_expected)
        plt.savefig(os.path.join(plot_dir, "loss_contour.png"))
        plt.close()
        print("Generated loss_contour.png")

        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(r_grid, m_grid, loss_data, cmap="viridis_r", edgecolor='none')
        fig.colorbar(surf, shrink=0.5, aspect=5, label='Average Loss')
        ax.set_title(f'Loss 3D Surface Plot (Nodes={nodes}, Epochs={epochs})') # Title will now reflect epoch=2
        ax.set_xlabel('Fuzzy R')
        ax.set_ylabel('Fuzzy M')
        ax.set_zlabel('Average Loss')
        ax.set_xticks(r_values_expected)
        ax.set_yticks(m_values_expected)
        plt.savefig(os.path.join(plot_dir, "loss_3d_surface.png"))
        plt.close()
        print("Generated loss_3d_surface.png")
    else:
        print("Skipping Loss plots due to missing data.")

    print("\nAnalysis complete.")


if __name__ == "__main__":
    # Use argparse to potentially make it more flexible later
    parser = argparse.ArgumentParser(description="Analyze Grid Search Results for Fuzzy Entropy AFL")
    # Default file name matches the one created by the shell script (updated for epoch=2)
    default_results_file = "gpu_grid_search_results_node8_epoch2.csv"
    parser.add_argument('--file', type=str, default=default_results_file,
                        help=f"Path to the grid search results CSV file (default: {default_results_file})")
    # Add arguments for nodes/epochs used in the search for verification (updated default epoch count)
    parser.add_argument('--nodes', type=int, default=8, help="Number of nodes used in the experiment (default: 8)")
    parser.add_argument('--epochs', type=int, default=2, help="Number of epochs used in the experiment (default: 2)")

    args = parser.parse_args()

    analyze_results(args.file, args.nodes, args.epochs)
