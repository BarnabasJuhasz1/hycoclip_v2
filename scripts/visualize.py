import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from functools import partial
from typing import List, Dict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hycoclip import lorentz as L


class VisualizationTools:

    def __init__(self, model, tokenizer, curv=None):
        self.model = model
        self.tokenizer = tokenizer
        self.curv = curv
        self.dataset_scores = {}
    
    def plot_distributions(self, *dists: List[Dict], filename:str = None, lines: List[Dict] = None):
        """
        Plots multiple distributions on the same figure.
        
        Args:
            *dists: Variable number of dictionaries containing distribution data.
                Each dictionary should have keys: 'title', 'data', 'xlabel', 'ylabel', 'color', and 'legend'.
        """
        fig, ax = plt.subplots(nrows=len(dists), figsize=(10, 8), constrained_layout=True)
        for dist, a in zip(dists, ax):
            sns.histplot(dist['data'], bins=30, alpha=0.5, color=dist['color'], edgecolor=dist['color'], label=dist['legend'], kde=True, ax=a)
            a.set_title(dist['title'])
            a.set_xlabel(dist['xlabel'])
            a.set_ylabel(dist['ylabel'])
            a.legend()

        if lines:
            if len(lines) == 1:
                lines = lines * len(ax) # Repeat the line for each subplot
            assert len(lines) == len(ax), "Number of lines must match number of subplots."
            for line, a in zip(lines, ax):
                for l in line:
                    a.axvline(l['x'], color=l['color'], linestyle='--')
                    a.text(l['x']-0.01*l['x'], l['y'], l['label'], color="black", fontsize=14, ha='center', va='bottom')

        plt.savefig(os.path.join("results", filename) if filename else "distribution_plot.png")
        plt.close(fig)        
        
    def visualize_probing_scores(self, dataset, dataset_name: str):
        assert dataset_name.lower() in self.probing_datasets, f"Dataset {dataset_name} not found in probing datasets."
        scores = self.get_dataset_scores(dataset, dataset_name, return_distances=True)
        
        # 1. visualize the distances between text pairs and image pairs
        dist_c = [s["distance"].c0_c1 for s in scores]
        dist_i = [s["distance"].i0_i1 for s in scores]
        distribution_1 = {"title": "Text Distances", "data": dist_c, "xlabel": "Distance", "ylabel": "Frequency", "color": "blue", "legend": "$\mathrm{d}(c_0, c_1)$"}
        distribution_2 = {"title": "Image Distances", "data": dist_i, "xlabel": "Distance", "ylabel": "Frequency", "color": "orange", "legend": "$\mathrm{d}(i_0, i_1)$"}
        self.plot_distributions(distribution_1, distribution_2, filename=f"{dataset_name}_distances.png")
        
        # 2. visualize the scores (right pair vs wrong pair)
        correct_scores = [(s["score"].c0_i0, s["score"].c1_i1) for s in scores]
        wrong_scores = [(s["score"].c0_i1, s["score"].c1_i0) for s in scores]
        # Average the scores for correct and wrong pairs
        avg_correct_scores = [sum(pair) / 2 for pair in correct_scores]
        avg_wrong_scores = [sum(pair) / 2 for pair in wrong_scores]
        scores_distribution_1 = {"title": "Avg. Correct Scores", "data": avg_correct_scores, "xlabel": "Score", "ylabel": "Frequency", "color": "green", "legend": "$\mathrm{avg}(\mathrm{score}(c_0, i_0),\mathrm{score}(c_1, i_1))$"}
        scores_distribution_2 = {"title": "Avg. Wrong Scores", "data": avg_wrong_scores, "xlabel": "Score", "ylabel": "Frequency", "color": "red", "legend": "$\mathrm{avg}(\mathrm{score}(c_0, i_1),\mathrm{score}(c_1, i_0))$"}
        self.plot_distributions(scores_distribution_1, scores_distribution_2, filename=f"{dataset_name}_avg_scores.png")
        # Difference between correct and wrong scores - Image
        diff_c0_i = [cs[0] - ws[0] for cs, ws in zip(correct_scores, wrong_scores)]
        diff_c1_i = [cs[1] - ws[1] for cs, ws in zip(correct_scores, wrong_scores)]
        diff_scores_distribution_1 = {"title": "Difference in Correct and Wrong Scores", "data": diff_c0_i, "xlabel": "Difference Score", "ylabel": "Frequency", "color": "purple", "legend": "$\mathrm{score}(c_0, i_0) - \mathrm{score}(c_0, i_1)$"}
        diff_scores_distribution_2 = {"title": "Difference in Correct and Wrong Scores", "data": diff_c1_i, "xlabel": "Difference Score", "ylabel": "Frequency", "color": "brown", "legend": "$\mathrm{score}(c_1, i_1) - \mathrm{score}(c_1, i_0)$"}
        self.plot_distributions(diff_scores_distribution_1, diff_scores_distribution_2, filename=f"{dataset_name}_diff_scores_image.png")
        # Difference between correct and wrong scores - Text
        diff_i0_c = [cs[0] - ws[1] for cs, ws in zip(correct_scores, wrong_scores)]
        diff_i1_c = [cs[1] - ws[0] for cs, ws in zip(correct_scores, wrong_scores)]
        diff_scores_distribution_3 = {"title": "Difference in Correct and Wrong Scores", "data": diff_i0_c, "xlabel": "Difference Score", "ylabel": "Frequency", "color": "cyan", "legend": "$\mathrm{score}(c_0, i_0) - \mathrm{score}(c_1, i_0)$"}
        diff_scores_distribution_4 = {"title": "Difference in Correct and Wrong Scores", "data": diff_i1_c, "xlabel": "Difference Score", "ylabel": "Frequency", "color": "magenta", "legend": "$\mathrm{score}(c_1, i_1) - \mathrm{score}(c_0, i_1)$"}
        self.plot_distributions(diff_scores_distribution_3, diff_scores_distribution_4, filename=f"{dataset_name}_diff_scores_text.png")
        # Scores for correct and wrong pairs
        correct_scores = [score for pair in correct_scores for score in pair]
        wrong_scores = [score for pair in wrong_scores for score in pair]
        scores_distribution_1 = {"title": "Correct Scores", "data": correct_scores, "xlabel": "Score", "ylabel": "Frequency", "color": "green", "legend": "$\mathrm{score}(c_0, i_0)$ and $\mathrm{score}(c_1, i_1)$"}
        scores_distribution_2 = {"title": "Wrong Scores", "data": wrong_scores, "xlabel": "Score", "ylabel": "Frequency", "color": "red", "legend": "$\mathrm{score}(c_0, i_1)$ and $\mathrm{score}(c_1, i_0)$"}
        self.plot_distributions(scores_distribution_1, scores_distribution_2, filename=f"{dataset_name}_scores.png")
        
        # 3. visualize the apertures
        aperture_c0 = [s["aperture"].c0 for s in scores]
        aperture_c1 = [s["aperture"].c1 for s in scores]
        aperture_i0 = [s["aperture"].i0 for s in scores]
        aperture_i1 = [s["aperture"].i1 for s in scores]
        aperture_distribution_1 = {"title": "Aperture Text", "data": aperture_c0 + aperture_c1, "xlabel": "Aperture", "ylabel": "Frequency", "color": "blue", "legend": "$\mathrm{Aperture}(c)$"}
        aperture_distribution_2 = {"title": "Aperture Image", "data": aperture_i0 + aperture_i1, "xlabel": "Aperture", "ylabel": "Frequency", "color": "green", "legend": "$\mathrm{Aperture}(i)$"}
        self.plot_distributions(aperture_distribution_1, aperture_distribution_2, filename=f"{dataset_name}_apertures.png", lines=[[
            {"x": np.arcsin(1 - 1e-5), "y": 0.1, "label": r"$\frac{\pi}{2}$", "color": "red", "linestyle": "--"},
        ]])
        
        # 4. visualize the norms
        norms_c0 = [s["norms"].c0 for s in scores]
        norms_c1 = [s["norms"].c1 for s in scores]
        norms_i0 = [s["norms"].i0 for s in scores]
        norms_i1 = [s["norms"].i1 for s in scores]
        norms_distribution_1 = {"title": "Norm Text", "data": norms_c0 + norms_c1, "xlabel": "Norm", "ylabel": "Frequency", "color": "purple", "legend": "$\mathrm{Norm}(c)$"}
        norms_distribution_2 = {"title": "Norm Image", "data": norms_i0 + norms_i1, "xlabel": "Norm", "ylabel": "Frequency", "color": "orange", "legend": "$\mathrm{Norm}(i)$"}
        self.plot_distributions(norms_distribution_1, norms_distribution_2, filename=f"{dataset_name}_norms.png", lines=[[
            {"x": 1.0, "y": 0.1, "label": "Norm = 1", "color": "black", "linestyle": "--"},
        ]])
        
        # 5. visualize the pairwise angles
        pairwise_angles_c = scores[-1]["pairwise_angles_c"]
        pairwise_angles_i = scores[-1]["pairwise_angles_i"]
        pairwise_angles_distribution_1 = {"title": "Pairwise Angles Text", "data": pairwise_angles_c, "xlabel": "Angle (degrees)", "ylabel": "Frequency", "color": "brown", "legend": "$\mathrm{Pairwise Angle}(c)$"}
        pairwise_angles_distribution_2 = {"title": "Pairwise Angles Image", "data": pairwise_angles_i, "xlabel": "Angle (degrees)", "ylabel": "Frequency", "color": "gray", "legend": "$\mathrm{Pairwise Angle}(i)$"}
        self.plot_distributions(pairwise_angles_distribution_1, pairwise_angles_distribution_2, filename=f"{dataset_name}_pairwise_angles.png")
        
        # 6. visualize the mean angles
        mean_angle_c = scores[-1]["mean_angle_c"]
        mean_angle_i = scores[-1]["mean_angle_i"]
        mean_angles_distribution_1 = {"title": "Angle with Mean - Text", "data": [mean_angle_c], "xlabel": "Angle (degrees)", "ylabel": "Frequency", "color": "cyan", "legend": "$\mathrm{Angle with Mean}(c)$"}
        mean_angles_distribution_2 = {"title": "Angle with Mean - Image", "data": [mean_angle_i], "xlabel": "Angle (degrees)", "ylabel": "Frequency", "color": "magenta", "legend": "$\mathrm{Angle with Mean}(i)$"}
        self.plot_distributions(mean_angles_distribution_1, mean_angles_distribution_2, filename=f"{dataset_name}_mean_angles.png")
        
        print(f"Visualizations saved for {dataset_name} dataset.")

    def plot_3d_function(self, func, xs: torch.Tensor, ys: torch.Tensor, filename: str = None, title: str = "3D Function Plot"):
        """
        Plot the estimated function with given parameters
        :param title: Tittle for the plot
        :param W: first layer weights
        :param b: bias
        :param v: output layer weights
        :param sigma: hyperparameter for tanh
        :return Show plot
        """
        fig = plt.figure(figsize=(12, 8))
        ax = plt.axes(projection='3d')

        X, Y = torch.meshgrid(xs, ys)
        XY = torch.column_stack([X.ravel(), Y.ravel()])
        Z = func(XY, XY).reshape(X.shape)
        ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_title(title)
        plt.savefig(os.path.join("results", filename) if filename else "3d_function_plot.png")
        plt.close(fig)
        print(f"3D function plot saved as {filename if filename else '3d_function_plot.png'}")

    def plot_3d_function_plotly(self, func, xs: torch.Tensor, ys: torch.Tensor, filename: str = None, title: str = "3D Function Plot"):
        """
        Plot the estimated function with given parameters using Plotly
        :param func: function to evaluate
        :param xs: x values (1D torch tensor)
        :param ys: y values (1D torch tensor)
        :param filename: optional filename to save (HTML)
        :param title: title for the plot
        """
        import plotly.graph_objects as go
        
        # Create meshgrid
        reference = torch.tensor([1.0, 1.0]).unsqueeze(0)  # Example reference point, adjust as needed
        X, Y = torch.meshgrid(xs, ys, indexing='ij')  # indexing='ij' to match numpy default
        XY = torch.column_stack([X.ravel(), Y.ravel()])
        Z = func(XY, reference).reshape(X.shape).detach().cpu().numpy()

        # Convert to numpy for Plotly
        X_np = X.detach().cpu().numpy()
        Y_np = Y.detach().cpu().numpy()

        # Create surface plot
        fig = go.Figure(data=[go.Surface(z=Z, x=X_np, y=Y_np, colorscale="Viridis")])
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title="x",
                yaxis_title="y",
                zaxis_title="z"
            ),
            autosize=True,
            width=900,
            height=700
        )

        # Save or show
        save_path = os.path.join("results", filename) if filename else "3d_function_plot.html"
        fig.write_html(save_path)
        print(f"3D function plot saved as {save_path}")
        
    def plot_2d_function_animated(self, func, xs: torch.Tensor, ys: torch.Tensor, filename: str = None, title: str = "2D Function Animation"):

        import matplotlib.animation as animation
        
        nframes = 50

        # --- Define grid ---
        X, Y = torch.meshgrid(xs, ys)
        XY = torch.column_stack([X.ravel(), Y.ravel()])
        x_lim = (X.min().item(), X.max().item())
        y_lim = (Y.min().item(), Y.max().item())
        Vx = torch.linspace(0, x_lim[1], nframes)

        V = torch.tensor([0.0, 0.0]).unsqueeze(0)  # Example vector, adjust as needed
        Z = func(V, XY).reshape(X.shape).detach().cpu().numpy()

        # --- Animation ---
        fig, ax = plt.subplots()
        c = ax.contourf(X, Y, Z, levels=30, cmap="viridis")  # start empty
        fig.colorbar(c)

        def animate(i):
            ax.clear()
            V = torch.tensor([Vx[i], 0.0]).unsqueeze(0)  # Example vector, adjust as needed
            Z = func(V, XY).reshape(X.shape).detach().cpu().numpy()
            c = ax.contourf(X, Y, Z, levels=30, cmap="viridis")
            ax.set_xlim(*x_lim)
            ax.set_ylim(*y_lim)
            return c.collections

        ani = animation.FuncAnimation(fig, animate, frames=nframes, interval=200, blit=False)

        plt.title(title)
        plt.xlabel('x')
        plt.ylabel('y')

        # Save or show
        save_path = filename if filename else "2d_function_animation.mp4"
        ani.save(save_path, writer='ffmpeg', fps=10)
        print(f"2D function animation saved as {save_path}")


    def plot_latent_space(self, x:np.ndarray=None, x_path:str=None, backend:str='plotly', title:str='Latent Space Visualization', filename:str=None):
        if x is None and x_path is None:
            raise ValueError("Either x or x_path must be provided.")

        if x_path is not None:
            x = np.load(x_path)
            
        x_keys = list(x.keys())
        print(f"Keys in the data: {x_keys}")
        assert all(k in x_keys for k in ['image_feats', 'box_image_feats', 'text_feats', 'box_text_feats']), "Data must contain keys: 'image_feats', 'box_image_feats', 'text_feats', 'box_text_feats'."
        
        # Stats for bounds
        all_feats = np.concatenate([x['image_feats'], x['box_image_feats'], x['text_feats'], x['box_text_feats']], axis=0)
        all_feats_min_x = all_feats[:, 0].min()
        all_feats_max_x = all_feats[:, 0].max()
        all_feats_min_y = all_feats[:, 1].min()
        all_feats_max_y = all_feats[:, 1].max()
        print(f"Feature bounds: x [{all_feats_min_x}, {all_feats_max_x}], y [{all_feats_min_y}, {all_feats_max_y}]")
        square_edge = max(abs(all_feats_min_x), abs(all_feats_max_x), abs(all_feats_min_y), abs(all_feats_max_y)) * 1.1

        colors = ['blue', 'red', 'orange', 'green']
        # Plotting code here
        if backend == 'matplotlib':
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.set_title(title)
            ax.set_xlim(-square_edge, square_edge)
            ax.set_ylim(-square_edge, square_edge)
            ax.set_aspect('equal')
            circle = plt.Circle((0, 0), square_edge, color='lightgray', alpha=0.5, fill=True)
            ax.add_artist(circle)
            for i, group in enumerate(x_keys):
                if "feats" in group:
                    sns.scatterplot(x=x[group][:, 0], y=x[group][:, 1], ax=ax, s=5, label=group, color=colors[i])
            ax.scatter(x=0, y=0, s=10, marker='x', color='black')
            plt.text(0.01, 0.01, 'O', fontsize=9)
            ax.set_axis_off()
            if filename:
                plt.savefig(filename)
            return fig, ax
        elif backend == 'plotly':
            import plotly.graph_objects as go
            fig = go.Figure()
            fig.update_layout(
                title=title,
                xaxis=dict(range=[-square_edge, square_edge], scaleanchor="y", scaleratio=1),
                yaxis=dict(range=[-square_edge, square_edge]),
                width=700,
                height=700,
                showlegend=True
            )
            # Add circle boundary
            theta = np.linspace(0, 2 * np.pi, 100)
            circle_x = square_edge * np.cos(theta)
            circle_y = square_edge * np.sin(theta)
            fig.add_trace(go.Scatter(x=circle_x, y=circle_y, mode='lines', line=dict(color='lightgray'), fill='toself', fillcolor='lightgray', opacity=0.5, showlegend=False))
            for i, group in enumerate(x_keys):
                if "feats" in group:
                    if "text" in group:
                        hover_info = [str(item) for item in x[group.replace("_feats", "")].flatten()]
                    else:
                        hover_info = [f"Image {j}" for j in range(x[group].shape[0])]
                    fig.add_trace(go.Scatter(x=x[group][:, 0], y=x[group][:, 1], mode='markers', name=group, marker=dict(size=5, color=colors[i]), opacity=0.7, 
                                            hoverinfo='text', hovertext=hover_info))
            # Add origin
            fig.add_trace(go.Scatter(x=[0], y=[0], mode='markers+text', marker=dict(size=10, color='black', symbol='x'), text=['O'], textposition='top right', name='Origin'))
            if filename:
                fig.write_html(filename)
                print(f"Latent space visualization saved as {filename}")
            return fig
            
            
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Visualize properties of hyperbolic functions and models.")
    parser.add_argument('--out-dir', type=str, default="results", help='Output directory for the visualizations')
    parser.add_argument('--filename', type=str, default="visualization.html", help='Output filename for the visualization')
    parser.add_argument('--fig-title', type=str, default="Hyperbolic Function Visualization", help='Title for the visualization')
    
    subparsers = parser.add_subparsers()
    
    # Add subparser for the hyperbolic functions
    parser_fns = subparsers.add_parser('fn', help='Visualize hyperbolic functions')
    # Add subparser for the models
    parser_models = subparsers.add_parser('model', help='Visualize model properties')
    
    parser_fns.add_argument('--fn', type=str, required=True, help='Function to visualize (e.g., "elementwise_dist", "oxy_angle")')
    parser_fns.add_argument('--curv', type=float, default=1.0, help='Curvature for hyperbolic functions')
    parser_fns.add_argument('--type', type=str, choices=['2d', '3d', '3d-interactive'], default='2d', help='Type of visualization (2d or 3d)')
    
    parser_models.add_argument('--x-path', type=str, required=True, help='Path to the .npz file containing embeddings')
    parser_models.add_argument('--backend', type=str, choices=['matplotlib', 'plotly'], default='plotly', help='Backend for visualization')
    
    args = parser.parse_args()

    if "fn" in args.__dict__:
        
        func_name = args.fn
        curv = torch.tensor(args.curv)
        
        if func_name == "hyp_dist":
            func = L.elementwise_dist
        elif func_name == "hyp_inner":
            func = L.elementwise_inner
        elif func_name == "hyp_dist_border":
            func = L.elementwise_distance_cone_border
        elif func_name == "hyp_chord":
            func = L.elementwise_chord_length
        elif func_name == "oxy_angle":
            func = L.oxy_angle
        elif func_name == "hyp_dist*oxy_angle":
            func = lambda x, y, curv: L.elementwise_dist(x, y, curv=curv) * L.oxy_angle(x, y, curv)
        else:
            raise ValueError(f"Unknown function: {func_name}.")
        
        func = partial(func, curv=curv)  # Example function, replace with your own
        xs = torch.linspace(-10, 10, 500)
        ys = torch.linspace(-10, 10, 500)
        
        viz_tools = VisualizationTools(None, None)  # No model needed for this example
        
        if args.type == '2d':
            viz_tools.plot_2d_function_animated(func, xs, ys, filename=os.path.join(args.out_dir, args.filename), title=args.fig_title)
        elif args.type == '3d':
            viz_tools.plot_3d_function(func, xs, ys, filename=os.path.join(args.out_dir, args.filename), title=args.fig_title)
        elif args.type == '3d-interactive':
            viz_tools.plot_3d_function_plotly(func, xs, ys, filename=os.path.join(args.out_dir, args.filename), title=args.fig_title)
    else:
        viz_tools = VisualizationTools(None, None)  # No model needed for this example
        viz_tools.plot_latent_space(x_path=args.x_path, backend=args.backend, title=args.fig_title, filename=os.path.join(args.out_dir, args.filename))