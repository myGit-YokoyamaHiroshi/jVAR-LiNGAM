# -*- coding: utf-8 -*-
"""
Created on Wed Sep 10 14:25:40 2025

@author: H.Yokoyama
"""
import numpy as np
import json, warnings, os, pathlib
import matplotlib
import networkx as nx
from matplotlib.colors import ListedColormap
import matplotlib.transforms as transforms
from matplotlib import pyplot, ticker
from matplotlib.ticker import FormatStrFormatter
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
from mpl_toolkits.axes_grid1 import make_axes_locatable
import sys
from operator import sub
import tigramite.data_processing as pp
from copy import deepcopy
import matplotlib.path as mpath
import matplotlib.patheffects as PathEffects
from mpl_toolkits.axisartist.axislines import Axes
import csv


def plot_graph(
    graph,
    val_matrix=None,
    var_names=None,
    fig_ax=None,
    figsize=None,
    save_name=None,
    link_colorbar_label="MCI",
    node_colorbar_label="auto-MCI",
    link_width=None,
    link_attribute=None,
    node_pos=None,
    arrow_linewidth=8.0,
    vmin_edges=-1,
    vmax_edges=1.0,
    edge_ticks=0.4,
    cmap_edges="RdBu_r",
    vmin_nodes=-1,
    vmax_nodes=1.0,
    node_ticks=0.4,
    cmap_nodes="RdBu_r",
    node_size=0.3,
    node_aspect=None,
    arrowhead_size=20,
    curved_radius=0.2,
    label_fontsize=10,
    tick_label_size=6,
    alpha=1.0,
    node_label_size=10,
    link_label_fontsize=10,
    lag_array=None,
    # network_lower_bound=0.2,
    show_colorbar=True,
    inner_edge_style="dashed",
    link_matrix=None,
    special_nodes=None,
    show_autodependency_lags=False
):
    """Creates a network plot.
    
    This is still in beta. The network is defined from links in graph. Nodes
    denote variables, straight links contemporaneous dependencies and curved
    arrows lagged dependencies. The node color denotes the maximal absolute
    auto-dependency and the link color the value at the lag with maximal
    absolute cross-dependency. The link label lists the lags with significant
    dependency in order of absolute magnitude. The network can also be
    plotted over a map drawn before on the same axis. Then the node positions
    can be supplied in appropriate axis coordinates via node_pos.

    Parameters
    ----------
    graph : string or bool array-like, optional (default: None)
        Either string matrix providing graph or bool array providing only adjacencies
        Must be of same shape as val_matrix. 
    val_matrix : array_like
        Matrix of shape (N, N, tau_max+1) containing test statistic values.
    var_names : list, optional (default: None)
        List of variable names. If None, range(N) is used.
    fig_ax : tuple of figure and axis object, optional (default: None)
        Figure and axes instance. If None they are created.
    figsize : tuple
        Size of figure.
    save_name : str, optional (default: None)
        Name of figure file to save figure. If None, figure is shown in window.
    link_colorbar_label : str, optional (default: 'MCI')
        Test statistic label.
    node_colorbar_label : str, optional (default: 'auto-MCI')
        Test statistic label for auto-dependencies.
    link_width : array-like, optional (default: None)
        Array of val_matrix.shape specifying relative link width with maximum
        given by arrow_linewidth. If None, all links have same width.
    link_attribute : array-like, optional (default: None)
        String array of val_matrix.shape specifying link attributes.
    node_pos : dictionary, optional (default: None)
        Dictionary of node positions in axis coordinates of form
        node_pos = {'x':array of shape (N,), 'y':array of shape(N)}. These
        coordinates could have been transformed before for basemap plots. You can
        also add a key 'transform':ccrs.PlateCarree() in order to plot graphs on 
        a map using cartopy.
    arrow_linewidth : float, optional (default: 30)
        Linewidth.
    vmin_edges : float, optional (default: -1)
        Link colorbar scale lower bound.
    vmax_edges : float, optional (default: 1)
        Link colorbar scale upper bound.
    edge_ticks : float, optional (default: 0.4)
        Link tick mark interval.
    cmap_edges : str, optional (default: 'RdBu_r')
        Colormap for links.
    vmin_nodes : float, optional (default: 0)
        Node colorbar scale lower bound.
    vmax_nodes : float, optional (default: 1)
        Node colorbar scale upper bound.
    node_ticks : float, optional (default: 0.4)
        Node tick mark interval.
    cmap_nodes : str, optional (default: 'OrRd')
        Colormap for links.
    node_size : int, optional (default: 0.3)
        Node size.
    node_aspect : float, optional (default: None)
        Ratio between the heigth and width of the varible nodes.
    arrowhead_size : int, optional (default: 20)
        Size of link arrow head. Passed on to FancyArrowPatch object.
    curved_radius, float, optional (default: 0.2)
        Curvature of links. Passed on to FancyArrowPatch object.
    label_fontsize : int, optional (default: 10)
        Fontsize of colorbar labels.
    alpha : float, optional (default: 1.)
        Opacity.
    node_label_size : int, optional (default: 10)
        Fontsize of node labels.
    link_label_fontsize : int, optional (default: 6)
        Fontsize of link labels.
    tick_label_size : int, optional (default: 6)
        Fontsize of tick labels.
    lag_array : array, optional (default: None)
        Optional specification of lags overwriting np.arange(0, tau_max+1)
    show_colorbar : bool
        Whether to show colorbars for links and nodes.
    show_autodependency_lags : bool (default: False)
        Shows significant autodependencies for a node.
    """

    if link_matrix is not None:
        raise ValueError("link_matrix is deprecated and replaced by graph array"
                         " which is now returned by all methods.")

    if fig_ax is None:
        fig = pyplot.figure(figsize=figsize)
        ax = fig.add_subplot(111, frame_on=False)
    else:
        fig, ax = fig_ax

    graph = np.copy(graph.squeeze())

    if graph.ndim == 4:
        raise ValueError("Time series graph of shape (N,N,tau_max+1,tau_max+1) cannot be represented by plot_graph,"
                         " use plot_time_series_graph instead.")

    if graph.ndim == 2:
        # If a non-time series (N,N)-graph is given, insert a dummy dimension
        graph = np.expand_dims(graph, axis = 2)

    if val_matrix is None:
        no_coloring = True
        cmap_edges = None
        cmap_nodes = None
    else:
        no_coloring = False

    
    N, N, dummy = graph.shape
    tau_max = dummy - 1
    max_lag = tau_max + 1

    if np.count_nonzero(graph != "") == np.count_nonzero(
        np.diagonal(graph) != ""
    ):
        diagonal = True
    else:
        diagonal = False

    if np.count_nonzero(graph == "") == graph.size or diagonal:
        graph[0, 1, 0] = "xxx"  # Workaround, will not be plotted... 
        no_links = True
    else:
        no_links = False

    if var_names is None:
        var_names = range(N)

    # Define graph links by absolute maximum (positive or negative like for
    # partial correlation)
    # val_matrix[np.abs(val_matrix) < sig_thres] = 0.

    # Only draw link in one direction among contemp
    # Remove lower triangle
    link_matrix_upper = np.copy(graph)
    link_matrix_upper[:, :, 0] = np.triu(link_matrix_upper[:, :, 0])

    # net = _get_absmax(link_matrix != "")
    net = np.any(link_matrix_upper != "", axis=2)
    G = nx.DiGraph(net)
    
    # This handels Graphs with no links.
    # nx.draw(G, alpha=0, zorder=-10)

    node_color = list(np.zeros(N))

    if show_autodependency_lags:
        autodep_sig_lags = np.full(N, None, dtype='object')
    else:
        autodep_sig_lags = None

    # list of all strengths for color map
    all_strengths = []
    # Add attributes, contemporaneous and lagged links are handled separately
    for (u, v, dic) in G.edges(data=True):
        dic["no_links"] = no_links
        # average lagfunc for link u --> v ANDOR u -- v
        if tau_max > 0:
            # argmax of absolute maximum where a link exists!
            links = np.where(link_matrix_upper[u, v, 1:] != "")[0]
            if len(links) > 0:
                argmax_links = np.abs(val_matrix[u, v][1:][links]).argmax()
                argmax = links[argmax_links] + 1
            else:
                argmax = 0
        else:
            argmax = 0

        if u != v:
            # For contemp links masking or finite samples can lead to different
            # values for u--v and v--u
            # Here we use the  maximum for the width and weight (=color)
            # of the link
            # Draw link if u--v OR v--u at lag 0 is nonzero
            # dic['inner_edge'] = ((np.abs(val_matrix[u, v][0]) >=
            #                       sig_thres[u, v][0]) or
            #                      (np.abs(val_matrix[v, u][0]) >=
            #                       sig_thres[v, u][0]))
            dic["inner_edge"] = link_matrix_upper[u, v, 0]
            dic["inner_edge_type"] = link_matrix_upper[u, v, 0]
            dic["inner_edge_alpha"] = alpha
            if no_coloring:
                dic["inner_edge_color"] = None
            else:
                dic["inner_edge_color"] = val_matrix[u, v, 0]
            # # value at argmax of average
            # if np.abs(val_matrix[u, v][0] - val_matrix[v, u][0]) > .0001:
            #     print("Contemporaneous I(%d; %d)=%.3f != I(%d; %d)=%.3f" % (
            #           u, v, val_matrix[u, v][0], v, u, val_matrix[v, u][0]) +
            #           " due to conditions, finite sample effects or "
            #           "masking, here edge color = "
            #           "larger (absolute) value.")
            # dic['inner_edge_color'] = _get_absmax(
            #     np.array([[[val_matrix[u, v][0],
            #                    val_matrix[v, u][0]]]])).squeeze()

            if link_width is None:
                dic["inner_edge_width"] = arrow_linewidth
            else:
                dic["inner_edge_width"] = (
                    link_width[u, v, 0] / link_width.max() * arrow_linewidth
                )

            if link_attribute is None:
                dic["inner_edge_attribute"] = None
            else:
                dic["inner_edge_attribute"] = link_attribute[u, v, 0]

            #     # fraction of nonzero values
            dic["inner_edge_style"] = "solid"
            # else:
            # dic['inner_edge_style'] = link_style[
            #         u, v, 0]

            all_strengths.append(dic["inner_edge_color"])

            if tau_max > 0:
                # True if ensemble mean at lags > 0 is nonzero
                # dic['outer_edge'] = np.any(
                #     np.abs(val_matrix[u, v][1:]) >= sig_thres[u, v][1:])
                dic["outer_edge"] = np.any(link_matrix_upper[u, v, 1:] != "")
            else:
                dic["outer_edge"] = False
            # print(u, v, dic["outer_edge"], argmax, link_matrix_upper[u, v, :])

            dic["outer_edge_type"] = link_matrix_upper[u, v, argmax]

            dic["outer_edge_alpha"] = alpha
            if link_width is None:
                # fraction of nonzero values
                dic["outer_edge_width"] = arrow_linewidth
            else:
                dic["outer_edge_width"] = (
                    link_width[u, v, argmax] / link_width.max() * arrow_linewidth
                )

            if link_attribute is None:
                # fraction of nonzero values
                dic["outer_edge_attribute"] = None
            else:
                dic["outer_edge_attribute"] = link_attribute[u, v, argmax]

            # value at argmax of average
            if no_coloring:
                dic["outer_edge_color"] = None
            else:
                dic["outer_edge_color"] = val_matrix[u, v][argmax]
            all_strengths.append(dic["outer_edge_color"])

            # Sorted list of significant lags (only if robust wrt
            # d['min_ensemble_frac'])
            if tau_max > 0:
                lags = np.abs(val_matrix[u, v][1:]).argsort()[::-1] + 1
                sig_lags = (np.where(link_matrix_upper[u, v, 1:] != "")[0] + 1).tolist()
            else:
                lags, sig_lags = [], []
            if lag_array is not None:
                dic["label"] = ",".join([str(lag_array[l]) for l in lags if l in sig_lags])  #str([str(lag_array[l]) for l in lags if l in sig_lags])[1:-1].replace(" ", "")
            else:
                dic["label"] = ",".join([str(l) for l in lags if l in sig_lags]) # str([str(l) for l in lags if l in sig_lags])[1:-1].replace(" ", "")
        else:
            # Node color is max of average autodependency
            if no_coloring:
                node_color[u] = None
            else:
                node_color[u] = val_matrix[u, v][argmax]

            if show_autodependency_lags:
                autodep_sig_lags[u] = "\n\n\n" + ",".join(str(i) for i in (np.where(link_matrix_upper[u, v, 1:] != "")[0] + 1).tolist())
                # Lags upto tau_max
                #autodep_lags = np.argsort(val_matrix[u, v][1:])[::-1]
                #autodep_lags += 1
                #autodeplags[u] = "\n\n\n" + ",".join(str(i) for i in autodep_lags.tolist())

            dic["inner_edge_attribute"] = None
            dic["outer_edge_attribute"] = None

        # dic['outer_edge_edge'] = False
        # dic['outer_edge_edgecolor'] = None
        # dic['inner_edge_edge'] = False
        # dic['inner_edge_edgecolor'] = None

    if special_nodes is not None:
        special_nodes_draw = {}
        for node in special_nodes:
            i, tau = node
            if tau >= -tau_max:
                special_nodes_draw[i] = special_nodes[node]
        special_nodes = special_nodes_draw
    

    # If no links are present, set value to zero
    if len(all_strengths) == 0:
        all_strengths = [0.0]

    if node_pos is None:
        pos = nx.circular_layout(deepcopy(G))
    else:
        pos = {}
        for i in range(N):
            pos[i] = (node_pos["x"][i], node_pos["y"][i])

    if node_pos is not None and 'transform' in node_pos: 
        transform = node_pos['transform']
    else: transform = ax.transData

    if cmap_nodes is None:
        node_color = None

    node_rings = {
        0: {
            "sizes": None,
            "color_array": node_color,
            "cmap": cmap_nodes,
            "vmin": vmin_nodes,
            "vmax": vmax_nodes,
            "ticks": node_ticks,
            "label": node_colorbar_label,
            "colorbar": show_colorbar,
        }
    }

    _draw_network_with_curved_edges(
        fig=fig,
        ax=ax,
        G=deepcopy(G),
        pos=pos,
        # dictionary of rings: {0:{'sizes':(N,)-array, 'color_array':(N,)-array
        # or None, 'cmap':string,
        node_rings=node_rings,
        # 'vmin':float or None, 'vmax':float or None, 'label':string or None}}
        node_labels=var_names,
        node_label_size=node_label_size,
        node_alpha=alpha,
        standard_size=node_size,
        node_aspect=node_aspect,
        standard_cmap="OrRd",
        standard_color_nodes="lightgrey",
        standard_color_links="black",
        log_sizes=False,
        cmap_links=cmap_edges,
        links_vmin=vmin_edges,
        links_vmax=vmax_edges,
        links_ticks=edge_ticks,
        tick_label_size=tick_label_size,
        # cmap_links_edges='YlOrRd', links_edges_vmin=-1., links_edges_vmax=1.,
        # links_edges_ticks=.2, link_edge_colorbar_label='link_edge',
        arrowstyle="simple",
        arrowhead_size=arrowhead_size,
        curved_radius=curved_radius,
        label_fontsize=label_fontsize,
        link_label_fontsize=link_label_fontsize,
        link_colorbar_label=link_colorbar_label,
        # network_lower_bound=network_lower_bound,
        show_colorbar=show_colorbar,
        # label_fraction=label_fraction,
        special_nodes=special_nodes,
        autodep_sig_lags=autodep_sig_lags,
        show_autodependency_lags=show_autodependency_lags,
        transform=transform
    )

    if save_name is not None:
        pyplot.savefig(save_name, dpi=300)
    else:
        return fig, ax
################################################################

def plot_time_series_graph(
        graph,
        val_matrix=None,
        var_names=None,
        fig_ax=None,
        figsize=None,
        link_colorbar_label="MCI",
        save_name=None,
        link_width=None,
        link_attribute=None,
        arrow_linewidth=4,
        vmin_edges=-1,
        vmax_edges=1.0,
        edge_ticks=0.4,
        cmap_edges="RdBu_r",
        order=None,
        node_size=0.1,
        node_aspect=None,
        arrowhead_size=20,
        curved_radius=0.2,
        label_fontsize=10,
        tick_label_size=6,
        alpha=1.0,
        inner_edge_style="dashed",
        link_matrix=None,
        special_nodes=None,
        node_classification=None,
        # aux_graph=None,
        standard_color_links='black',
        standard_color_nodes='lightgrey',
):
    """Creates a time series graph.
    This is still in beta. The time series graph's links are colored by
    val_matrix.

    Parameters
    ----------
    graph : string or bool array-like, optional (default: None)
        Either string matrix providing graph or bool array providing only adjacencies
        Either of shape (N, N, tau_max + 1) or as auxiliary graph of dims 
        (N, N, tau_max+1, tau_max+1) describing auxADMG. 
    val_matrix : array_like
        Matrix of same shape as graph containing test statistic values.
    var_names : list, optional (default: None)
        List of variable names. If None, range(N) is used.
    fig_ax : tuple of figure and axis object, optional (default: None)
        Figure and axes instance. If None they are created.
    figsize : tuple
        Size of figure.
    save_name : str, optional (default: None)
        Name of figure file to save figure. If None, figure is shown in window.
    link_colorbar_label : str, optional (default: 'MCI')
        Test statistic label.
    link_width : array-like, optional (default: None)
        Array of val_matrix.shape specifying relative link width with maximum
        given by arrow_linewidth. If None, all links have same width.
    link_attribute : array-like, optional (default: None)
        Array of graph.shape specifying specific in drawing the graph (for internal use).
    order : list, optional (default: None)
        order of variables from top to bottom.
    arrow_linewidth : float, optional (default: 30)
        Linewidth.
    vmin_edges : float, optional (default: -1)
        Link colorbar scale lower bound.
    vmax_edges : float, optional (default: 1)
        Link colorbar scale upper bound.
    edge_ticks : float, optional (default: 0.4)
        Link tick mark interval.
    cmap_edges : str, optional (default: 'RdBu_r')
        Colormap for links.
    node_size : int, optional (default: 0.1)
        Node size.
    node_aspect : float, optional (default: None)
        Ratio between the heigth and width of the varible nodes.
    arrowhead_size : int, optional (default: 20)
        Size of link arrow head. Passed on to FancyArrowPatch object.
    curved_radius, float, optional (default: 0.2)
        Curvature of links. Passed on to FancyArrowPatch object.
    label_fontsize : int, optional (default: 10)
        Fontsize of colorbar labels.
    alpha : float, optional (default: 1.)
        Opacity.
    node_label_size : int, optional (default: 10)
        Fontsize of node labels.
    link_label_fontsize : int, optional (default: 6)
        Fontsize of link labels.
    tick_label_size : int, optional (default: 6)
        Fontsize of tick labels.
    inner_edge_style : string, optional (default: 'dashed')
        Style of inner_edge contemporaneous links.
    special_nodes : dict
        Dictionary of format {(i, -tau): 'blue', ...} to color special nodes.
    node_classification : dict or None (default: None)
        Dictionary of format {i: 'space_context', ...} to classify nodes into system, context, or dummy nodes.
        Keys of the dictionary are from {0, ..., N-1} where N is the number of nodes.
        Options for the values are "system", "time_context", "space_context", "time_dummy", or "space_dummy".
        Space_contexts and dummy nodes need to be represented as a single node in the time series graph.
        In case no value is supplied all nodes are treated as system nodes, i.e. are plotted in a time-resolved manner.
    """

    if link_matrix is not None:
        raise ValueError("link_matrix is deprecated and replaced by graph array"
                         " which is now returned by all methods.")
        
    if fig_ax is None:
        fig = pyplot.figure(figsize=figsize)
        ax = fig.add_subplot(111, frame_on=False)
    else:
        fig, ax = fig_ax

    if val_matrix is None:
        no_coloring = True
        cmap_edges = None
    else:
        no_coloring = False



    if graph.ndim == 4:
        N, N, dummy, _ = graph.shape
        tau_max = dummy - 1
        max_lag = tau_max + 1
    else:
        N, N, dummy = graph.shape
        tau_max = dummy - 1
        max_lag = tau_max + 1

    if np.count_nonzero(graph == "") == graph.size:
        if graph.ndim == 4:
            graph[0, 1, 0, 0] = "---"
        else:
            graph[0, 1, 0] = "---"
        no_links = True
    else:
        no_links = False

    if var_names is None:
        var_names = range(N)

    if order is None:
        order = range(N)

    if set(order) != set(range(N)):
        raise ValueError("order must be a permutation of range(N)")

    def translate(row, lag):
        return row * max_lag + lag

    # Define graph links by absolute maximum (positive or negative like for
    # partial correlation)
    tsg = np.zeros((N * max_lag, N * max_lag))
    tsg_val = np.zeros((N * max_lag, N * max_lag))
    tsg_width = np.zeros((N * max_lag, N * max_lag))
    tsg_style = np.zeros((N * max_lag, N * max_lag), dtype=graph.dtype)
    if link_attribute is not None:
        tsg_attr = np.zeros((N * max_lag, N * max_lag), dtype=link_attribute.dtype)

    if graph.ndim == 4:
        # 4-dimensional graphs represent the finite-time window projection of stationary 3-d graphs
        # They are internally created in some classes
        # Only draw link in one direction
        for i, j, taui, tauj in np.column_stack(np.where(graph)):
            tau = taui - tauj
            # if tau <= 0 and j <= i:
            if translate(i,   max_lag - 1 - taui) >= translate(j, max_lag-1-tauj):
                continue
            # print(max_lag, (i, -taui), (j, -tauj), aux_graph[i, j, taui, tauj])
            # print(translate(i, max_lag - 1 - taui), translate(j, max_lag-1-tauj))
            tsg[translate(i,   max_lag - 1 - taui), translate(j, max_lag-1-tauj)] = 1.0
            tsg_val[translate(i,   max_lag - 1 - taui), translate(j, max_lag-1-tauj)] = val_matrix[i, j, taui, tauj]
            tsg_style[translate(i,   max_lag - 1 - taui), translate(j, max_lag-1-tauj)] = graph[i, j, taui, tauj]
            if link_width is not None:
                tsg_width[translate(i,   max_lag - 1 - taui), translate(j, max_lag-1-tauj)] = link_width[i, j, taui, tauj] / link_width.max() * arrow_linewidth
            if link_attribute is not None:
                tsg_attr[translate(i,   max_lag - 1 - taui), translate(j, max_lag-1-tauj)] = link_attribute[i, j, taui, tauj] #'spurious'
        # print(tsg_style)   
            # print(tsg_style[translate(i,   max_lag - 1 - taui), translate(j, max_lag-1-tauj)] = graph[i, j, taui, tauj])    
            # print(max_lag, (i, -taui), (j, -tauj), graph[i, j, taui, tauj], tsg_style[translate(i,   max_lag - 1 - taui), translate(j, max_lag-1-tauj)])
 

    else:
      # Only draw link in one direction
      # Remove lower triangle
      link_matrix_tsg = np.copy(graph)
      link_matrix_tsg[:, :, 0] = np.triu(graph[:, :, 0])

      for i, j, tau in np.column_stack(np.where(link_matrix_tsg)):
        for t in range(max_lag):
            if (
                0 <= translate(i, t - tau)
                and translate(i, t - tau) % max_lag <= translate(j, t) % max_lag
            ):

                tsg[
                    translate(i, t - tau), translate(j, t)
                ] = 1.0  # val_matrix[i, j, tau]
                tsg_val[translate(i, t - tau), translate(j, t)] = val_matrix[i, j, tau]
                tsg_style[translate(i, t - tau), translate(j, t)] = graph[
                    i, j, tau
                ]
                if link_width is not None:
                    tsg_width[translate(i, t - tau), translate(j, t)] = (
                        link_width[i, j, tau] / link_width.max() * arrow_linewidth
                    )
                if link_attribute is not None:
                    tsg_attr[translate(i, t - tau), translate(j, t)] = link_attribute[
                        i, j, tau
                    ]


    G = nx.DiGraph(tsg)

    if special_nodes is not None:
        special_nodes_tsg = {}
        for node in special_nodes:
            i, tau = node
            if tau >= -tau_max:
                special_nodes_tsg[translate(i, max_lag-1 + tau)] = special_nodes[node]

        special_nodes = special_nodes_tsg

    if node_classification is None:
        node_classification = {i: "system" for i in range(N)}
    node_classification_tsg = {}
    for node in node_classification:
        for tau in range(max_lag):
            if tau == 0:
                suffix = "_first"
            elif tau == max_lag-1:
                suffix = "_last"
            else:
                suffix = "_middle"
            node_classification_tsg[translate(node, tau)] = node_classification[node] + suffix

    # node_color = np.zeros(N)
    # list of all strengths for color map
    all_strengths = []
    # Add attributes, contemporaneous and lagged links are handled separately
    for (u, v, dic) in G.edges(data=True):
        dic["no_links"] = no_links
        if u != v:
            # tau = np.abs((u - v) % max_lag)
            # Determine neighbors in TSG
            i = u // max_lag
            taui = -(max_lag -1 - (u % max_lag))
            j = v // max_lag
            tauj = -(max_lag -1 - (v % max_lag))

            if np.abs(i-j) <= 1 and np.abs(tauj-taui) <= 1:
                inout = 'inner'
                dic["inner_edge"] = True
                dic["outer_edge"] = False
            else:
                inout = 'outer'
                dic["inner_edge"] = False
                dic["outer_edge"] = True

            dic["%s_edge_type" % inout] = tsg_style[u, v]

            dic["%s_edge_alpha" % inout] = alpha

            if link_width is None:
                # fraction of nonzero values
                dic["%s_edge_width" % inout] = dic["%s_edge_width" % inout] = arrow_linewidth
            else:
                dic["%s_edge_width" % inout] = dic["%s_edge_width" % inout] = tsg_width[u, v]

            if link_attribute is None:
                dic["%s_edge_attribute" % inout] = None
            else:
                dic["%s_edge_attribute" % inout] = tsg_attr[u, v]

            # value at argmax of average
            if no_coloring:
                dic["%s_edge_color" % inout] = None
            else:
                dic["%s_edge_color" % inout] = tsg_val[u, v]

            all_strengths.append(dic["%s_edge_color" % inout])
            dic["label"] = None
        # print(u, v, dic)

    # If no links are present, set value to zero
    if len(all_strengths) == 0:
        all_strengths = [0.0]

    posarray = np.zeros((N * max_lag, 2))
    for i in range(N * max_lag):
        posarray[i] = np.array([(i % max_lag), (1.0 - i // max_lag)])

    pos_tmp = {}
    for i in range(N * max_lag):
        # for n in range(N):
        #     for tau in range(max_lag):
        #         i = n*N + tau
        pos_tmp[i] = np.array(
            [
                ((i % max_lag) - posarray.min(axis=0)[0])
                / (posarray.max(axis=0)[0] - posarray.min(axis=0)[0]),
                ((1.0 - i // max_lag) - posarray.min(axis=0)[1])
                / (posarray.max(axis=0)[1] - posarray.min(axis=0)[1]),
            ]
        )
        pos_tmp[i][np.isnan(pos_tmp[i])] = 0.0

    pos = {}
    for n in range(N):
        for tau in range(max_lag):
            pos[n * max_lag + tau] = pos_tmp[order[n] * max_lag + tau]

    node_rings = {
        0: {"sizes": None, "color_array": None, "label": "", "colorbar": False,}
    }

    node_labels = ["" for i in range(N * max_lag)]

    if graph.ndim == 4 and val_matrix is None:
        show_colorbar = False
    else:
        show_colorbar = True

    _draw_network_with_curved_edges(
        fig=fig,
        ax=ax,
        G=deepcopy(G),
        pos=pos,
        node_rings=node_rings,
        node_labels=node_labels,
        # node_label_size=node_label_size,
        node_alpha=alpha,
        standard_size=node_size,
        node_aspect=node_aspect,
        standard_cmap="OrRd",
        standard_color_nodes=standard_color_nodes,
        standard_color_links=standard_color_links,
        log_sizes=False,
        cmap_links=cmap_edges,
        links_vmin=vmin_edges,
        links_vmax=vmax_edges,
        links_ticks=edge_ticks,
        # link_label_fontsize=link_label_fontsize,
        arrowstyle="simple",
        arrowhead_size=arrowhead_size,
        curved_radius=curved_radius,
        label_fontsize=label_fontsize,
        tick_label_size=tick_label_size,
        label_fraction=0.5,
        link_colorbar_label=link_colorbar_label,
        inner_edge_curved=False,
        # network_lower_bound=network_lower_bound,
        # network_left_bound=label_space_left,
        inner_edge_style=inner_edge_style,
        special_nodes=special_nodes,
        show_colorbar=show_colorbar,
        node_classification=node_classification_tsg,
        max_lag=max_lag,
    )

    for i in range(N):
        trans = transforms.blended_transform_factory(ax.transAxes, ax.transData)
        # trans = transforms.blended_transform_factory(fig.transFigure, ax.transData)
        ax.text(
            0.,
            pos[order[i] * max_lag][1],
            f"{var_names[order[i]]}",
            fontsize=label_fontsize,
            horizontalalignment="right",
            verticalalignment="center",
            transform=trans,
        )

    for tau in np.arange(max_lag - 1, -1, -1):
        trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
        # trans = transforms.blended_transform_factory(ax.transData, fig.transFigure)
        if tau == max_lag - 1:
            ax.text(
                pos[tau][0],
                1.0, # - label_space_top,
                r"$t$",
                fontsize=int(label_fontsize * 1.2),
                horizontalalignment="center",
                verticalalignment="bottom",
                transform=trans,
            )
        else:
            ax.text(
                pos[tau][0],
                1.0, # - label_space_top,
                r"$t-%s$" % str(max_lag - tau - 1),
                fontsize=int(label_fontsize * 1.2),
                horizontalalignment="center",
                verticalalignment="bottom",
                transform=trans,
            )

    # pyplot.tight_layout()
    if save_name is not None:
        pyplot.savefig(save_name, dpi=300)
    else:
        return fig, ax

    
################################################################


def _draw_network_with_curved_edges(
        fig,
        ax,
        G,
        pos,
        node_rings,
        node_labels,
        node_label_size=10,
        node_alpha=1.0,
        standard_size=100,
        node_aspect=None,
        standard_cmap="OrRd",
        standard_color_links='black',
        standard_color_nodes='lightgrey',
        log_sizes=False,
        cmap_links="YlOrRd",
        # cmap_links_edges="YlOrRd",
        links_vmin=0.0,
        links_vmax=1.0,
        links_edges_vmin=0.0,
        links_edges_vmax=1.0,
        links_ticks=0.2,
        links_edges_ticks=0.2,
        link_label_fontsize=8,
        arrowstyle="->, head_width=0.4, head_length=1",
        arrowhead_size=3.0,
        curved_radius=0.2,
        label_fontsize=4,
        label_fraction=0.5,
        link_colorbar_label="link",
        tick_label_size=6,
        # link_edge_colorbar_label='link_edge',
        inner_edge_curved=False,
        inner_edge_style="solid",
        # network_lower_bound=0.2,
        network_left_bound=None,
        show_colorbar=True,
        special_nodes=None,
        autodep_sig_lags=None,
        show_autodependency_lags=False,
        transform='data',
        node_classification=None,
        max_lag=0,
):
    """Function to draw a network from networkx graph instance.
    Various attributes are used to specify the graph's properties.
    This function is just a beta-template for now that can be further
    customized.
    """

    if transform == 'data':
        transform = ax.transData

    from matplotlib.patches import FancyArrowPatch, Circle, Ellipse

    ax.spines["left"].set_color("none")
    ax.spines["right"].set_color("none")
    ax.spines["bottom"].set_color("none")
    ax.spines["top"].set_color("none")
    ax.set_xticks([])
    ax.set_yticks([])

    N = len(G)

    # This fixes a positioning bug in matplotlib.
    ax.scatter(0, 0, zorder=-10, alpha=0)

    def draw_edge(
        ax,
        u,
        v,
        d,
        seen,
        arrowstyle= "Simple, head_width=2, head_length=2, tail_width=1",
        outer_edge=True,
    ):

        # avoiding attribute error raised by changes in networkx
        if hasattr(G, "node"):
            # works with networkx 1.10
            n1 = G.node[u]["patch"]
            n2 = G.node[v]["patch"]
        else:
            # works with networkx 2.4
            n1 = G.nodes[u]["patch"]
            n2 = G.nodes[v]["patch"]

        # print("+++++++++++++++++++++++==cmap_links ", cmap_links)
        if outer_edge:
            rad = -1.0 * curved_radius
            if cmap_links is not None:
                facecolor = data_to_rgb_links.to_rgba(d["outer_edge_color"])
            else:
                if d["outer_edge_color"] is not None:
                    facecolor = d["outer_edge_color"]
                else:
                    facecolor = standard_color_links

            width = d["outer_edge_width"]
            alpha = d["outer_edge_alpha"]
            if (u, v) in seen:
                rad = seen.get((u, v))
                rad = (rad + np.sign(rad) * 0.1) * -1.0
            arrowstyle = arrowstyle
            # link_edge = d['outer_edge_edge']
            linestyle = 'solid' # d.get("outer_edge_style")

            if d.get("outer_edge_attribute", None) == "spurious":
                facecolor = "grey"

            if d.get("outer_edge_type") in ["<-o", "<--", "<-x", "<-+"]:
                n1, n2 = n2, n1

            if d.get("outer_edge_type") in [
                "o-o",
                "o--",
                "--o",
                "---",
                "x-x",
                "x--",
                "--x",
                "o-x",
                "x-o",
                # "+->",
                # "<-+",
            ]:
                arrowstyle = "-"
                # linewidth = width*factor
            elif d.get("outer_edge_type") == "<->":
                # arrowstyle = "<->, head_width=0.4, head_length=1"
                arrowstyle = "Simple, head_width=2, head_length=2, tail_width=1" #%float(width/20.)
            elif d.get("outer_edge_type") in ["o->", "-->", "<-o", "<--", "<-x", "x->", "+->", "<-+"]:
                # arrowstyle = "->, head_width=0.4, head_length=1"
                # arrowstyle = "->, head_width=0.4, head_length=1, width=10"
                arrowstyle = "Simple, head_width=2, head_length=2, tail_width=1" #%float(width/20.)
            else:
                arrowstyle = "Simple, head_width=2, head_length=2, tail_width=1" #%float(width/20.)
                # raise ValueError("edge type %s not valid." %d.get("outer_edge_type"))
        else:
            rad = -1.0 * inner_edge_curved * curved_radius
            if cmap_links is not None:
                facecolor = data_to_rgb_links.to_rgba(d["inner_edge_color"])
            else:
                if d["inner_edge_color"] is not None:
                    facecolor = d["inner_edge_color"]
                else:
                    # print("HERE")
                    facecolor = standard_color_links

            width = d["inner_edge_width"]
            alpha = d["inner_edge_alpha"]

            if d.get("inner_edge_attribute", None) == "spurious":
                facecolor = "grey"
            # print(d.get("inner_edge_type"))
            if d.get("inner_edge_type") in ["<-o", "<--", "<-x", "<-+"]:
                n1, n2 = n2, n1

            if d.get("inner_edge_type") in [
                "o-o",
                "o--",
                "--o",
                "---",
                "x-x",
                "x--",
                "--x",
                "o-x",
                "x-o",
            ]:
                arrowstyle = "-"
            elif d.get("inner_edge_type") == "<->":
                # arrowstyle = "<->, head_width=0.4, head_length=1"
                arrowstyle = "Simple, head_width=2, head_length=2, tail_width=1" #%float(width/20.)
            elif d.get("inner_edge_type") in ["o->", "-->", "<-o", "<--", "<-x", "x->", "+->", "<-+"]:
                # arrowstyle = "->, head_width=0.4, head_length=1"
                arrowstyle = "Simple, head_width=2, head_length=2, tail_width=1" #%float(width/20.)
            else:
                arrowstyle = "Simple, head_width=2, head_length=2, tail_width=1" #%float(width/20.)

            #     raise ValueError("edge type %s not valid." %d.get("inner_edge_type"))

            linestyle = 'solid' #d.get("inner_edge_style")

        coor1 = n1.center
        coor2 = n2.center

        marker_size = width ** 2
        figuresize = fig.get_size_inches()

        # print("COLOR ", facecolor)
        # print(u, v, outer_edge, "outer ", d.get("outer_edge_type"),  "inner ",  d.get("inner_edge_type"), width, arrowstyle, linestyle)
        
        if ((outer_edge is True and d.get("outer_edge_type") == "<->")
           or (outer_edge is False and d.get("inner_edge_type") == "<->")):
            e_p = FancyArrowPatch(
                coor1,
                coor2,
                arrowstyle=arrowstyle,
                connectionstyle=f"arc3,rad={rad}",
                mutation_scale=1*width,
                lw=0., #width / 2.,
                aa=True,
                alpha=alpha,
                linestyle=linestyle,
                color=facecolor,
                clip_on=False,
                patchA=n1,
                patchB=n2,
                shrinkA=7,
                shrinkB=0,
                zorder=-1,
                capstyle="butt",
                transform=transform,
            )
            ax.add_artist(e_p)

            e_p_back = FancyArrowPatch(
              coor2,
              coor1,
              arrowstyle=arrowstyle,
              connectionstyle=f"arc3,rad={-rad}",
              mutation_scale=1*width,
              lw=0., #width / 2.,
              aa=True,
              alpha=alpha,
              linestyle=linestyle,
              color=facecolor,
              clip_on=False,
              patchA=n2,
              patchB=n1,
              shrinkA=7,
              shrinkB=0,
              zorder=-1,
              capstyle="butt",
              transform=transform,
            )  
            ax.add_artist(e_p_back)

        else:
            if arrowstyle == '-':
                lw = 1*width
            else:
                lw = 0.
            # e_p = FancyArrowPatch(
            #     coor1,
            #     coor2,
            #     arrowstyle=arrowstyle,
            #     connectionstyle=f"arc3,rad={rad}",
            #     mutation_scale=np.sqrt(width)*2*1.1,
            #     lw=lw*1.1, #width / 2.,
            #     aa=True,
            #     alpha=alpha,
            #     linestyle=linestyle,
            #     color='white',
            #     clip_on=False,
            #     patchA=n1,
            #     patchB=n2,
            #     shrinkA=0,
            #     shrinkB=0,
            #     zorder=-1,
            #     capstyle="butt",
            # )
            # ax.add_artist(e_p)
            e_p = FancyArrowPatch(
                coor1,
                coor2,
                arrowstyle=arrowstyle,
                connectionstyle=f"arc3,rad={rad}",
                mutation_scale=1*width,
                lw=lw, #width / 2.,
                aa=True,
                alpha=alpha,
                linestyle=linestyle,
                color=facecolor,
                clip_on=False,
                patchA=n1,
                patchB=n2,
                shrinkA=0,
                shrinkB=0,
                # zorder=-1,
                capstyle="butt",
                transform=transform,
            )
            ax.add_artist(e_p)

        e_p_marker = FancyArrowPatch(
                coor1,
                coor2,
                arrowstyle='-',
                connectionstyle=f"arc3,rad={rad}",
                mutation_scale=1*width,
                lw=0., #width / 2.,
                aa=True,
                alpha=0.,
                linestyle=linestyle,
                color=facecolor,
                clip_on=False,
                patchA=n1,
                patchB=n2,
                shrinkA=0,
                shrinkB=0,
                zorder=-10,
                capstyle="butt",
                transform=transform,
        )
        ax.add_artist(e_p_marker)

        # marker_path = e_p_marker.get_path()
        vertices = e_p_marker.get_path().vertices.copy()
        # vertices = e_p_marker.get_verts()
        # vertices = e_p_marker.get_path().to_polygons(transform=None)[0]
        # print(vertices.shape)
        m, n = vertices.shape

        # print(vertices)
        start = vertices[0]
        end = vertices[-1]

        # This must be added to avoid rescaling of the plot, when no 'o'
        # or 'x' is added to the graph.
        ax.scatter(*start, zorder=-10, alpha=0, transform=transform,)

        if outer_edge:
            if d.get("outer_edge_type") in ["o->", "o--"]:
                circle_marker_start = ax.scatter(
                    *start,
                    marker="o",
                    s=marker_size,
                    facecolor="w",
                    edgecolor=facecolor,
                    zorder=1,
                    transform=transform,
                )
                ax.add_collection(circle_marker_start)
            elif d.get("outer_edge_type") == "<-o":
                circle_marker_end = ax.scatter(
                    *start,
                    marker="o",
                    s=marker_size,
                    facecolor="w",
                    edgecolor=facecolor,
                    zorder=1,
                    transform=transform,
                )
                ax.add_collection(circle_marker_end)
            elif d.get("outer_edge_type") == "--o":
                circle_marker_end = ax.scatter(
                    *end,
                    marker="o",
                    s=marker_size,
                    facecolor="w",
                    edgecolor=facecolor,
                    zorder=1,
                    transform=transform,
                )
                ax.add_collection(circle_marker_end)
            elif d.get("outer_edge_type") in ["x--", "x->"]:
                circle_marker_start = ax.scatter(
                    *start,
                    marker="X",
                    s=marker_size,
                    facecolor="w",
                    edgecolor=facecolor,
                    zorder=1,
                    transform=transform,
                )
                ax.add_collection(circle_marker_start)
            elif d.get("outer_edge_type") in ["+--", "+->"]:
                circle_marker_start = ax.scatter(
                    *start,
                    marker="P",
                    s=marker_size,
                    facecolor="w",
                    edgecolor=facecolor,
                    zorder=1,
                    transform=transform,
                )
                ax.add_collection(circle_marker_start)
            elif d.get("outer_edge_type") == "<-x":
                circle_marker_end = ax.scatter(
                    *start,
                    marker="X",
                    s=marker_size,
                    facecolor="w",
                    edgecolor=facecolor,
                    zorder=1,
                    transform=transform,
                )
                ax.add_collection(circle_marker_end)
            elif d.get("outer_edge_type") == "<-+":
                circle_marker_end = ax.scatter(
                    *start,
                    marker="P",
                    s=marker_size,
                    facecolor="w",
                    edgecolor=facecolor,
                    zorder=1,
                    transform=transform,
                )
                ax.add_collection(circle_marker_end)
            elif d.get("outer_edge_type") == "--x":
                circle_marker_end = ax.scatter(
                    *end,
                    marker="X",
                    s=marker_size,
                    facecolor="w",
                    edgecolor=facecolor,
                    zorder=1,
                    transform=transform,
                )
                ax.add_collection(circle_marker_end)
            elif d.get("outer_edge_type") == "o-o":
                circle_marker_start = ax.scatter(
                    *start,
                    marker="o",
                    s=marker_size,
                    facecolor="w",
                    edgecolor=facecolor,
                    zorder=1,
                    transform=transform,
                )
                ax.add_collection(circle_marker_start)
                circle_marker_end = ax.scatter(
                    *end,
                    marker="o",
                    s=marker_size,
                    facecolor="w",
                    edgecolor=facecolor,
                    zorder=1,
                    transform=transform,
                )
                ax.add_collection(circle_marker_end)
            elif d.get("outer_edge_type") == "x-x":
                circle_marker_start = ax.scatter(
                    *start,
                    marker="X",
                    s=marker_size,
                    facecolor="w",
                    edgecolor=facecolor,
                    zorder=1,
                    transform=transform,
                )
                ax.add_collection(circle_marker_start)
                circle_marker_end = ax.scatter(
                    *end,
                    marker="X",
                    s=marker_size,
                    facecolor="w",
                    edgecolor=facecolor,
                    zorder=1,
                    transform=transform,
                )
                ax.add_collection(circle_marker_end)
            elif d.get("outer_edge_type") == "o-x":
                circle_marker_start = ax.scatter(
                    *start,
                    marker="o",
                    s=marker_size,
                    facecolor="w",
                    edgecolor=facecolor,
                    zorder=1,
                    transform=transform,
                )
                ax.add_collection(circle_marker_start)
                circle_marker_end = ax.scatter(
                    *end,
                    marker="X",
                    s=marker_size,
                    facecolor="w",
                    edgecolor=facecolor,
                    zorder=1,
                    transform=transform,
                )
                ax.add_collection(circle_marker_end)
            elif d.get("outer_edge_type") == "x-o":
                circle_marker_start = ax.scatter(
                    *start,
                    marker="X",
                    s=marker_size,
                    facecolor="w",
                    edgecolor=facecolor,
                    zorder=1,
                    transform=transform,
                )
                ax.add_collection(circle_marker_start)
                circle_marker_end = ax.scatter(
                    *end,
                    marker="o",
                    s=marker_size,
                    facecolor="w",
                    edgecolor=facecolor,
                    zorder=1,
                    transform=transform,
                )
                ax.add_collection(circle_marker_end)

        else:
            if d.get("inner_edge_type") in ["o->", "o--"]:
                circle_marker_start = ax.scatter(
                    *start,
                    marker="o",
                    s=marker_size,
                    facecolor="w",
                    edgecolor=facecolor,
                    zorder=1,
                    transform=transform,
                )
                ax.add_collection(circle_marker_start)
            elif d.get("inner_edge_type") == "<-o":
                circle_marker_end = ax.scatter(
                    *start,
                    marker="o",
                    s=marker_size,
                    facecolor="w",
                    edgecolor=facecolor,
                    zorder=1,
                    transform=transform,
                )
                ax.add_collection(circle_marker_end)
            elif d.get("inner_edge_type") == "--o":
                circle_marker_end = ax.scatter(
                    *end,
                    marker="o",
                    s=marker_size,
                    facecolor="w",
                    edgecolor=facecolor,
                    zorder=1,
                    transform=transform,
                )
                ax.add_collection(circle_marker_end)
            elif d.get("inner_edge_type") in ["x--", "x->"]:
                circle_marker_start = ax.scatter(
                    *start,
                    marker="X",
                    s=marker_size,
                    facecolor="w",
                    edgecolor=facecolor,
                    zorder=1,
                    transform=transform,
                )
                ax.add_collection(circle_marker_start)
            elif d.get("inner_edge_type") in ["+--", "+->"]:
                circle_marker_start = ax.scatter(
                    *start,
                    marker="P",
                    s=marker_size,
                    facecolor="w",
                    edgecolor=facecolor,
                    zorder=1,
                    transform=transform,
                )
                ax.add_collection(circle_marker_start)
            elif d.get("inner_edge_type") == "<-x":
                circle_marker_end = ax.scatter(
                    *start,
                    marker="X",
                    s=marker_size,
                    facecolor="w",
                    edgecolor=facecolor,
                    zorder=1,
                    transform=transform,
                )
                ax.add_collection(circle_marker_end)
            elif d.get("inner_edge_type") == "<-+":
                circle_marker_end = ax.scatter(
                    *start,
                    marker="P",
                    s=marker_size,
                    facecolor="w",
                    edgecolor=facecolor,
                    zorder=1,
                    transform=transform,
                )
                ax.add_collection(circle_marker_end)
            elif d.get("inner_edge_type") == "--x":
                circle_marker_end = ax.scatter(
                    *end,
                    marker="X",
                    s=marker_size,
                    facecolor="w",
                    edgecolor=facecolor,
                    zorder=1,
                    transform=transform,
                )
                ax.add_collection(circle_marker_end)
            elif d.get("inner_edge_type") == "o-o":
                circle_marker_start = ax.scatter(
                    *start,
                    marker="o",
                    s=marker_size,
                    facecolor="w",
                    edgecolor=facecolor,
                    zorder=1,
                    transform=transform,
                )
                ax.add_collection(circle_marker_start)
                circle_marker_end = ax.scatter(
                    *end,
                    marker="o",
                    s=marker_size,
                    facecolor="w",
                    edgecolor=facecolor,
                    zorder=1,
                    transform=transform,
                )
                ax.add_collection(circle_marker_end)
            elif d.get("inner_edge_type") == "x-x":
                circle_marker_start = ax.scatter(
                    *start,
                    marker="X",
                    s=marker_size,
                    facecolor="w",
                    edgecolor=facecolor,
                    zorder=1,
                    transform=transform,
                )
                ax.add_collection(circle_marker_start)
                circle_marker_end = ax.scatter(
                    *end,
                    marker="X",
                    s=marker_size,
                    facecolor="w",
                    edgecolor=facecolor,
                    zorder=1,
                    transform=transform,
                )
                ax.add_collection(circle_marker_end)
            elif d.get("inner_edge_type") == "o-x":
                circle_marker_start = ax.scatter(
                    *start,
                    marker="o",
                    s=marker_size,
                    facecolor="w",
                    edgecolor=facecolor,
                    zorder=1,
                    transform=transform,
                )
                ax.add_collection(circle_marker_start)
                circle_marker_end = ax.scatter(
                    *end,
                    marker="X",
                    s=marker_size,
                    facecolor="w",
                    edgecolor=facecolor,
                    zorder=1,
                    transform=transform,
                )
                ax.add_collection(circle_marker_end)
            elif d.get("inner_edge_type") == "x-o":
                circle_marker_start = ax.scatter(
                    *start,
                    marker="X",
                    s=marker_size,
                    facecolor="w",
                    edgecolor=facecolor,
                    zorder=1,
                    transform=transform,
                )
                ax.add_collection(circle_marker_start)
                circle_marker_end = ax.scatter(
                    *end,
                    marker="o",
                    s=marker_size,
                    facecolor="w",
                    edgecolor=facecolor,
                    zorder=1,
                    transform=transform,
                )
                ax.add_collection(circle_marker_end)



        if d["label"] is not None and outer_edge:
            def closest_node(node, nodes):
                nodes = np.asarray(nodes)
                node = node.reshape(1, 2)
                dist_2 = np.sum((nodes - node)**2, axis=1)
                return np.argmin(dist_2)

            # Attach labels of lags
            # trans = None  # patch.get_transform()
            # path = e_p.get_path()
            vertices = e_p_marker.get_path().vertices.copy()
            verts = e_p.get_path().to_polygons(transform=None)[0]
            # print(verts)
            # print(verts.shape)
            # print(vertices.shape)
            # for num, vert in enumerate(verts):
            #     ax.text(vert[0], vert[1], str(num), 
            #         transform=transform,)
            # ax.scatter(verts[:,0], verts[:,1])
            # mid_point = np.array([(start[0] + end[0])/2., (start[1] + end[1])/2.])
            # print(start, end, mid_point)
            # ax.scatter(mid_point[0], mid_point[1], marker='x', 
            #     s=100, zorder=10, transform=transform,)
            closest_node = closest_node(vertices[int(len(vertices)/2.),:], verts)
            # print(closest_node, verts[closest_node])
            # ax.scatter(verts[closest_node][0], verts[closest_node][1], marker='x')

            if len(vertices) > 2:
                # label_vert = vertices[int(len(vertices)/2.),:] #verts[1, :]
                label_vert = verts[closest_node] #verts[1, :]
                l = d["label"]
                string = str(l)
                txt = ax.text(
                    label_vert[0],
                    label_vert[1],
                    string,
                    fontsize=link_label_fontsize,
                    verticalalignment="center",
                    horizontalalignment="center",
                    color="w",
                    zorder=1,
                    transform=transform,
                )
                txt.set_path_effects(
                    [PathEffects.withStroke(linewidth=2, foreground="k")]
                )

        return rad

    # Collect all edge weights to get color scale
    all_links_weights = []
    all_links_edge_weights = []
    for (u, v, d) in G.edges(data=True):
        if u != v:
            if d["outer_edge"] and d["outer_edge_color"] is not None:
                all_links_weights.append(d["outer_edge_color"])
            if d["inner_edge"] and d["inner_edge_color"] is not None:
                all_links_weights.append(d["inner_edge_color"])

    if cmap_links is not None and len(all_links_weights) > 0:
        if links_vmin is None:
            links_vmin = np.array(all_links_weights).min()
        if links_vmax is None:
            links_vmax = np.array(all_links_weights).max()
        data_to_rgb_links = pyplot.cm.ScalarMappable(
            norm=None, cmap=pyplot.get_cmap(cmap_links)
        )
        data_to_rgb_links.set_array(np.array(all_links_weights))
        data_to_rgb_links.set_clim(vmin=links_vmin, vmax=links_vmax)
        # Create colorbars for links

        # setup colorbar axes.
        if show_colorbar:
            # cax_e = pyplot.axes(
            #     [
            #         0.55,
            #         ax.get_subplotspec().get_position(ax.figure).bounds[1] + 0.02,
            #         0.4,
            #         0.025 + (len(all_links_edge_weights) == 0) * 0.035,
            #     ],
            #     frameon=False,
            # )
            bbox_ax = ax.get_position()
            width = bbox_ax.xmax-bbox_ax.xmin
            height = bbox_ax.ymax-bbox_ax.ymin
            # print(bbox_ax.xmin, bbox_ax.xmax, bbox_ax.ymin, bbox_ax.ymax) 
            # cax_e = fig.add_axes(
            #     [
            #         bbox_ax.xmax - width*0.45,
            #         bbox_ax.ymin-0.075*height+network_lower_bound-0.15,
            #         width*0.4,
            #         0.075*height,   #0.025 + (len(all_links_edge_weights) == 0) * 0.035,
            #     ],
            #     frameon=False,
            # )
            cax_e = ax.inset_axes( 
                          [
                          0.55, -0.07, 0.4, 0.07
                    # bbox_ax.xmax - width*0.45,
                    # bbox_ax.ymin-0.075*height+network_lower_bound-0.15,
                    # width*0.4,
                    # 0.075*height,   #0.025 + (len(all_links_edge_weights) == 0) * 0.035,
                ],
                frameon=False,)
            # divider = make_axes_locatable(ax)

            # cax_e = divider.append_axes('bottom', size='5%', pad=0.05, frameon=False,)

            cb_e = pyplot.colorbar(
                data_to_rgb_links, cax=cax_e, orientation="horizontal"
            )
            # try:
            ticks_here = np.arange(
                    _myround(links_vmin, links_ticks, "down"),
                    _myround(links_vmax, links_ticks, "up") + links_ticks,
                    links_ticks,
                )
            cb_e.set_ticks(ticks_here[(links_vmin <= ticks_here) & (ticks_here <= links_vmax)])
            # except:
            #     print('no ticks given')

            cb_e.outline.clear()
            cax_e.set_xlabel(
                link_colorbar_label, labelpad=1, fontsize=label_fontsize, zorder=10
            )
            cax_e.tick_params(axis='both', which='major', labelsize=tick_label_size)

    ##
    # Draw nodes
    ##
    node_sizes = np.zeros((len(node_rings), N))
    for ring in list(node_rings):  # iterate through to get all node sizes
        if node_rings[ring]["sizes"] is not None:
            node_sizes[ring] = node_rings[ring]["sizes"]

        else:
            node_sizes[ring] = standard_size
    max_sizes = node_sizes.max(axis=1)
    total_max_size = node_sizes.sum(axis=0).max()
    node_sizes /= total_max_size
    node_sizes *= standard_size

    def get_aspect(ax):
        # Total figure size
        figW, figH = ax.get_figure().get_size_inches()
        # print(figW, figH)
        # Axis size on figure
        _, _, w, h = ax.get_position().bounds
        # Ratio of display units
        # print(w, h)
        disp_ratio = (figH * h) / (figW * w)
        # Ratio of data units
        # Negative over negative because of the order of subtraction
        data_ratio = sub(*ax.get_ylim()) / sub(*ax.get_xlim())
        # print(data_ratio, disp_ratio)
        return disp_ratio / data_ratio

    if node_aspect is None:
        node_aspect = get_aspect(ax)

    # start drawing the outer ring first...
    for ring in list(node_rings)[::-1]:
        #        print ring
        # dictionary of rings: {0:{'sizes':(N,)-array, 'color_array':(N,)-array
        # or None, 'cmap':string, 'vmin':float or None, 'vmax':float or None}}
        if node_rings[ring]["color_array"] is not None:
            color_data = node_rings[ring]["color_array"]
            if node_rings[ring]["vmin"] is not None:
                vmin = node_rings[ring]["vmin"]
            else:
                vmin = node_rings[ring]["color_array"].min()
            if node_rings[ring]["vmax"] is not None:
                vmax = node_rings[ring]["vmax"]
            else:
                vmax = node_rings[ring]["color_array"].max()
            if node_rings[ring]["cmap"] is not None:
                cmap = node_rings[ring]["cmap"]
            else:
                cmap = standard_cmap
            data_to_rgb = pyplot.cm.ScalarMappable(
                norm=None, cmap=pyplot.get_cmap(cmap)
            )
            data_to_rgb.set_array(color_data)
            data_to_rgb.set_clim(vmin=vmin, vmax=vmax)
            colors = [data_to_rgb.to_rgba(color_data[n]) for n in G]

            if node_rings[ring]["colorbar"]:
                # Create colorbars for nodes
                # cax_n = pyplot.axes([.8 + ring*0.11,
                # ax.get_subplotspec().get_position(ax.figure).bounds[1]+0.05, 0.025, 0.35], frameon=False) #
                # setup colorbar axes.
                # setup colorbar axes.
                bbox_ax = ax.get_position()
                # print(bbox_ax.xmin, bbox_ax.xmax, bbox_ax.ymin, bbox_ax.ymax) 
                cax_n = ax.inset_axes(
                    [
                    0.05, -0.07, 0.4, 0.07
                        # bbox_ax.xmin + width*0.05,
                        # bbox_ax.ymin-0.075*height+network_lower_bound-0.15,
                        # width*0.4,
                        # 0.075*height,   #0.025 + (len(all_links_edge_weights) == 0) * 0.035,
                    ],
                    frameon=False,
                )
                cb_n = pyplot.colorbar(data_to_rgb, cax=cax_n, orientation="horizontal")
                # try:
                ticks_here = np.arange(
                    _myround(vmin, node_rings[ring]["ticks"], "down"),
                    _myround(vmax, node_rings[ring]["ticks"], "up")
                    + node_rings[ring]["ticks"],
                    node_rings[ring]["ticks"],
                )
                cb_n.set_ticks(ticks_here[(vmin <= ticks_here) & (ticks_here <= vmax)])
                # except:
                #     print ('no ticks given')
                cb_n.outline.clear()
                # cb_n.set_ticks()
                cax_n.set_xlabel(
                    node_rings[ring]["label"], labelpad=1, fontsize=label_fontsize
                )
                cax_n.tick_params(axis='both', which='major', labelsize=tick_label_size)
        else:
            colors = None
            vmin = None
            vmax = None

        for n in G:
            if type(node_alpha) == dict:
                alpha = node_alpha[n]
            else:
                alpha = 1.0

            if special_nodes is not None:
                if n in special_nodes:
                    color_here = special_nodes[n]
                else:
                    color_here = 'grey'
            else:
                if colors is None:
                    color_here = standard_color_nodes
                else:
                    color_here = colors[n]

            c = Ellipse(
                pos[n],
                width=node_sizes[: ring + 1].sum(axis=0)[n] * node_aspect,
                height=node_sizes[: ring + 1].sum(axis=0)[n],
                clip_on=False,
                facecolor=color_here,
                edgecolor=color_here,
                zorder=-ring - 1 + 2,
                transform=transform,
            )

            # else:
            #     if special_nodes is not None and n in special_nodes:
            #         color_here = special_nodes[n]
            #     else:
            #         color_here = colors[n]
            #     c = Ellipse(
            #         pos[n],
            #         width=node_sizes[: ring + 1].sum(axis=0)[n] * node_aspect,
            #         height=node_sizes[: ring + 1].sum(axis=0)[n],
            #         clip_on=False,
            #         facecolor=colors[n],
            #         edgecolor=colors[n],
            #         zorder=-ring - 1,
            #     )

            ax.add_patch(c)

            if node_classification is not None and node_classification[n] in ["space_context_last", "space_dummy_last", "time_dummy_last"]:
                node_height = node_sizes[: ring + 1].sum(axis=0)[n]
                node_width_difference_to_height = node_height * (1 - node_aspect)

                c_wide = mpatches.FancyBboxPatch((pos[n-max_lag+1][0] + node_width_difference_to_height / 2, pos[n-max_lag+1][1]),
                                                 (pos[n][0] - pos[n-max_lag+1][0] - node_width_difference_to_height),
                                                 0.,
                                                 boxstyle=mpatches.BoxStyle.Round(pad=0.5 * node_height),
                                                 facecolor=color_here,
                                                 edgecolor=color_here,
                                                 )

                ax.add_patch(c_wide)


            # avoiding attribute error raised by changes in networkx
            if hasattr(G, "node"):
                # works with networkx 1.10
                G.node[n]["patch"] = c
            else:
                # works with networkx 2.4
                G.nodes[n]["patch"] = c

            if ring == 0:
                ax.text(
                    pos[n][0],
                    pos[n][1],
                    node_labels[n],
                    fontsize=node_label_size,
                    horizontalalignment="center",
                    verticalalignment="center",
                    alpha=1.0,
                    zorder=5.,
                    transform=transform,
                )
                if show_autodependency_lags:
                    ax.text(
                        pos[n][0],
                        pos[n][1],
                        autodep_sig_lags[n],
                        fontsize=link_label_fontsize,
                        horizontalalignment="center",
                        verticalalignment="center",
                        color="black",
                        zorder=5.,
                        transform=transform,
                    )

    # Draw edges
    seen = {}
    for (u, v, d) in G.edges(data=True):
        if d.get("no_links"):
            d["inner_edge_alpha"] = 1e-8
            d["outer_edge_alpha"] = 1e-8
        if u != v:
            if d["outer_edge"]:
                seen[(u, v)] = draw_edge(ax, u, v, d, seen, outer_edge=True)
            if d["inner_edge"]:
                seen[(u, v)] = draw_edge(ax, u, v, d, seen, outer_edge=False)

    # if network_left_bound is not None:
    #     network_right_bound = 0.98
    # else:
    #     network_right_bound = None
    # fig.subplots_adjust(bottom=network_lower_bound, left=network_left_bound, right=network_right_bound) #, right=0.97)


def _myround(x, base=5, round_mode="updown"):
    """Rounds x to a float with precision base."""

    if round_mode == "updown":
        return base * round(float(x) / base)
    elif round_mode == "down":
        return base * np.floor(float(x) / base)
    elif round_mode == "up":
        return base * np.ceil(float(x) / base)

    return base * round(float(x) / base)