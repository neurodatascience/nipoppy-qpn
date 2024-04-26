import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from nilearn import plotting
import pandas as pd
# import networkx as nx
import os
import warnings

warnings.simplefilter('ignore')

##########################################################################################

fig_dpi = 120
fig_bbox_inches = 'tight'
fig_pad = 0.1
show_title = False
save_fig_format = 'png'

##########################################################################################

######### functions #########

def plot_FC(
        FC,
        roi_labels=None,
        title='',
        reorder=False,
        save_image=False, output_root=None
    ):
    '''
    
    '''
    figsize = (10, 8)

    plotting.plot_matrix(
        FC, figure=figsize, labels=roi_labels,
        vmax=1, vmin=-1,
        reorder=reorder
    )

    if show_title:
        plt.suptitle(title, fontsize=15)

    if save_image:
        folder = output_root[:output_root.rfind('/')]
        if not os.path.exists(folder):
            os.makedirs(folder)
        plt.savefig(output_root+title+'.'+save_fig_format, 
            dpi=fig_dpi, bbox_inches=fig_bbox_inches, pad_inches=fig_pad, format=save_fig_format
        ) 
        plt.close()
    else:
        plt.show()

def cat_plot(data, 
    x=None, y=None, 
    hue=None,
    title='',
    save_image=False, output_root=None
    ):
    '''
    data is a dictionary with different vars as keys 
    '''

    sns.set_context("paper", 
        font_scale=1.0, 
        rc={"lines.linewidth": 1.0}
    )

    n_columns = len(data)

    fig_width = n_columns * 6
    fig_height = 6
    fig, axs = plt.subplots(1, n_columns, figsize=(fig_width, fig_height), 
        facecolor='w', edgecolor='k', sharex=False, sharey=False)

    for i, key in enumerate(data):

        df = pd.DataFrame(data[key])
        sns.violinplot(ax=axs[i], data=df, x=x, y=y, hue=hue, width=0.5, split=True)
        sns.stripplot(ax=axs[i], data=df, x=x, y=y, hue=hue, alpha=0.25, color='black')
        
        axs[i].set(xlabel=None)
        axs[i].set(ylabel=None)
        axs[i].set_title(key)
    
    if show_title:
        plt.suptitle(title, fontsize=15)

    if save_image:
        folder = output_root[:output_root.rfind('/')]
        if not os.path.exists(folder):
            os.makedirs(folder)
        plt.savefig(output_root+title+'.'+save_fig_format, 
            dpi=fig_dpi, bbox_inches=fig_bbox_inches, pad_inches=fig_pad, format=save_fig_format
        ) 
        plt.close()
    else:
        plt.show()
    
def pairwise_cat_plots(data, x, y, label=None,
    title='', 
    save_image=False, output_root=None
    ):
    '''
    data is a dictionary with different vars as keys 
    if label is specidied, it will be used as hue
    '''

    sns.set_context("paper", 
        font_scale=3.0, 
        rc={"lines.linewidth": 3.0}
    )

    row_keys = [key for key in data]
    n_rows = len(row_keys)
    column_keys = [key for key in data[row_keys[-1]]]
    n_columns = len(column_keys)

    sns.set_style('darkgrid')

    fig_width = n_columns * 6
    fig_height = n_rows * 6
    fig, axs = plt.subplots(n_rows, n_columns, figsize=(fig_width, fig_height), 
        facecolor='w', edgecolor='k', sharex=True, sharey=True)
    
    axs_plotted = list()
    for i, key_i in enumerate(data):
        for j, key_j in enumerate(data[key_i]):
            df = pd.DataFrame(data[key_i][key_j])

            gfg = sns.violinplot(ax=axs[i, j], data=df, x=x, y=y, hue=label, split=True)
            sns.stripplot(ax=axs[i, j], data=df, x=x, y=y, alpha=0.25, color='black')

            gfg.legend(
                # bbox_to_anchor= (1.2,1), 
                fontsize=20
            )
            axs[i, j].set(xlabel=None)
            axs[i, j].set(ylabel=None)
            axs[i, j].set_title(key_i+'-'+key_j)
            axs_plotted.append(axs[i, j])

    # remove extra subplots
    for ax in axs.ravel():
        if not ax in axs_plotted:
            ax.set_axis_off()
            ax.xaxis.set_tick_params(which='both', labelbottom=True)
    
    if show_title:
        plt.suptitle(title, fontsize=15, y=0.90)

    if save_image:
        folder = output_root[:output_root.rfind('/')]
        if not os.path.exists(folder):
            os.makedirs(folder)
        plt.savefig(output_root+title+'.'+save_fig_format, \
            dpi=fig_dpi, bbox_inches=fig_bbox_inches, pad_inches=fig_pad, format=save_fig_format \
        ) 
        plt.close()
    else:
        plt.show()

def find_network_label(node, networks):
    '''
    find the network that appears the node's name
    node is a string containing a node's name
    networks is a list of networks names
    '''
    for network in networks:
        if network in node:
            return network
    return None

def segment_FC(FC, nodes, networks):
    '''
    average FC values over each large network in
    networks
    the output FC matrix will be in the same order as 
    networks
    '''
    segmented = np.zeros((len(networks), len(networks)))
    counts = np.zeros((len(networks), len(networks)))
    for i, node_i in enumerate(nodes):
        network_i = networks.index(find_network_label(node_i, networks))
        for j, node_j in enumerate(nodes):
            network_j = networks.index(find_network_label(node_j, networks))
            segmented[network_i, network_j] += FC[i, j]
            counts[network_i, network_j] += 1
    return np.divide(segmented, counts, out=np.zeros_like(segmented), where=counts!=0) 

def FC2dict(FC_lst, networks, labels):

    output = {}
    for idx, FC in enumerate(FC_lst):
        
        for i, network_i in enumerate(networks):
            for j, network_j in enumerate(networks):

                if j>i:
                    continue

                if not network_i in output:
                    output[network_i] = {}
                if not network_j in output[network_i]:
                    output[network_i][network_j] = {'FC':list(), '':list(), 'label':list()}
            
                output[network_i][network_j]['FC'].append(FC[i, j])
                output[network_i][network_j][''].append('FC')
                output[network_i][network_j]['label'].append(labels[idx])
    return output

def calc_graph_propoerty(A, property, threshold=None, binarize=False):
    """
    calc_graph_propoerty: Computes Graph-based properties 
    of adjacency matrix A
    A is converted to positive before calc
    property:
        - ECM: Computes Eigenvector Centrality Mapping (ECM) 
        - shortest_path
        - degree
        - clustering_coef
        - communicability

    Input:

        A (np.array): adjacency matrix (must be >0)

    Output:

        graph-property (np.array): a vector
    """    

    G = nx.from_numpy_matrix(np.abs(A)) 
    G.remove_edges_from(nx.selfloop_edges(G))
    # G = G.to_undirected()

    # pruning edges 
    if not threshold is None:
        labels = [d["weight"] for (u, v, d) in G.edges(data=True)]
        labels.sort()
        ebunch = [(u, v) for u, v, d in G.edges(data=True) if d['weight']<threshold]
        G.remove_edges_from(ebunch)

    if binarize:
        weight='None'
    else:
        weight='weight'

    graph_property = None
    if property=='ECM':
        graph_property = nx.eigenvector_centrality(G, weight=weight)
        graph_property = [graph_property[node] for node in graph_property]
        graph_property = np.array(graph_property)
    if property=='shortest_path':
        SHORTEST_PATHS = dict(nx.shortest_path_length(G, weight=weight))
        graph_property = np.zeros((A.shape[0], A.shape[0]))
        for node_i in SHORTEST_PATHS:
            for node_j in SHORTEST_PATHS[node_i]:
                graph_property[node_i, node_j] = SHORTEST_PATHS[node_i][node_j]
        graph_property = graph_property + graph_property.T
        graph_property = graph_property[np.triu_indices(graph_property.shape[1], k=1)]
    if property=='degree':
        graph_property = [G.degree(weight=weight)[node] for node in G]
        graph_property = np.array(graph_property)
    if property=='clustering_coef':
        graph_property = nx.clustering(G, weight=weight)
        graph_property = [graph_property[node] for node in graph_property]
        graph_property = np.array(graph_property)
    if property=='communicability':
        comm = nx.communicability(G)
        graph_property = np.zeros((len(comm), len(comm)))
        for node_i in comm:
            for node_j in comm[node_i]:
                graph_property[node_i, node_j] = comm[node_i][node_j]
        graph_property = graph_property + graph_property.T
        graph_property = graph_property[np.triu_indices(graph_property.shape[1], k=1)]

    return graph_property

##########################################################################################

### paths to files
# local
root = '/Users/mte/Documents/McGill/JB/QPN/data/'
output_root = '/Users/mte/Documents/McGill/JB/QPN/result_outputs/'

# # server
# root = '../../../../pd/qpn/derivatives/fmriprep/v20.2.7/fmriprep/'
# output_root = '../outputs/FC_outputs/'

save_image = True

### parameters

reorder_conn_mat = False
visualize = False
brain_atlas = 'schaefer' # schaefer or seitzman
confound_strategy = 'no_motion_no_gsr' # no_motion or no_motion_no_gsr

##########################################################################################

# calc average static FC

metric = 'correlation' # correlation , covariance , precision 
dir = './FC_outputs/'
YEO_networks = ['Vis', 'SomMot', 'DorsAttn', 'SalVentAttn', 'Limbic', 'Cont','Default']

# load description and demographics
manifest = pd.read_csv('./mr_proc_manifest.csv')

ALL_RECORDS = os.listdir(dir)
ALL_RECORDS = [i for i in ALL_RECORDS if 'FC_output' in i]
ALL_RECORDS.sort()
print(str(len(ALL_RECORDS))+' subjects were found.')

# prepare the roi labels for visualization
FC = np.load(dir+ALL_RECORDS[0], allow_pickle='TRUE').item()
roi_labels = FC['roi_labels']
roi_labels = [str(label) for label in roi_labels]
roi_labels = [label[label.find('Networks')+9:-3] for label in roi_labels]

FC_lst= list()
FC_segmented_lst = list()
conditions = list()
bids_id_lst = [id for id in manifest['bids_id']]
for idx, subj in enumerate(ALL_RECORDS):
    bids_id = subj[:subj.find('_FC_output.npy')] 
    if bids_id in bids_id_lst: # if the subject id is not in the manifest, it will be excluded
        FC = np.load(dir+subj,allow_pickle='TRUE').item()
        segmented_FC = segment_FC(FC[metric], nodes=roi_labels, networks=YEO_networks)
        FC_lst.append(FC[metric])
        FC_segmented_lst.append(segmented_FC)
        conditions.append(manifest['group'][bids_id_lst.index(bids_id)])
    
print(str(conditions.count('CTRL'))+' CTRL subjects were found.')
print(str(conditions.count('PD'))+' PD subjects were found.')
print(str(len(ALL_RECORDS) - conditions.count('CTRL') - conditions.count('PD'))+' subjects were excluded.')

plot_FC(
    FC=np.mean(np.array(FC_lst), axis=0),
    roi_labels=roi_labels,
    title='average FC',
    reorder=reorder_conn_mat,
    save_image=save_image, output_root=output_root
)

plot_FC(
    FC=np.mean(np.array(FC_segmented_lst), axis=0),
    roi_labels=YEO_networks,
    title='segmented average FC',
    reorder=reorder_conn_mat,
    save_image=save_image, output_root=output_root
)

FC_dict = FC2dict(FC_lst=FC_segmented_lst, networks=YEO_networks, labels=conditions)

pairwise_cat_plots(FC_dict, x='', y='FC', label='label',
    title='FC_dist', 
    save_image=save_image, output_root=output_root
    )

## graph

title = 'graph properties'
RESULTS = {}

for threshold in [0.8]:
    for i, property in enumerate(['degree', 'communicability', 'shortest_path', 'clustering_coef']):
        
        RESULTS[property] = {'values':list(), 'condition':list(), '':list()}
        for j, FC in enumerate(FC_lst):
            features = calc_graph_propoerty(
                A=FC, 
                property=property, 
                threshold=threshold, 
                binarize=False
                )

            if property=='communicability' and np.mean(features)>2000:
                continue
            RESULTS[property][''].append('')
            RESULTS[property]['values'].append(np.mean(features))
            RESULTS[property]['condition'].append(conditions[j])    

    cat_plot(data=RESULTS, x='', y='values',
        hue='condition',
        title=title+'_threshold_'+str(threshold),
        save_image=save_image, output_root=output_root
        )

