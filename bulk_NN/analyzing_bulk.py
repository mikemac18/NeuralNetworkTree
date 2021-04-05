import numpy as np

bulk_acc = np.genfromtxt('bulkNN_accuracies.tsv', delimiter='\t')
bulk_batch = np.genfromtxt('num_batch.tsv', delimiter='\t')
bulk_epoch = np.genfromtxt('num_epochs.tsv', delimiter='\t')
bulk_num_layers = np.genfromtxt('bulk_layers.tsv', delimiter='\t')
bulk_nodes = np.genfromtxt('bulk_nodes.tsv', delimiter='\t')
input_nodes = np.genfromtxt('input_nodes.tsv', delimiter='\t')

acc_indeces = np.zeros(6)
acc_max = np.zeros(6)
bulk_b_index = np.zeros(6)
bulk_e_index = np.zeros(6)
bulk_nl_index = np.zeros(6)
bulk_nodes_index = np.zeros(6)
bulk_node_input_index = np.zeros(6)
acc_list = bulk_acc.tolist()

for i in range(len(acc_indeces)):
    max_num = max(acc_list)
    max_index = acc_list.index(max_num)
    acc_indeces[i] = max_index
    acc_max[i] = max_num
    acc_list.remove(max_num)
    bulk_b = bulk_batch[max_index]
    bulk_e = bulk_epoch[max_index]
    bulk_nl = bulk_num_layers[max_index]
    bulk_nodes = bulk_nodes_index[max_index]
    bulk_node_input = bulk_node_input_index[max_index]
    bulk_b_index[i] = bulk_b
    bulk_e_index[i] = bulk_e
    bulk_nl_index[i] = bulk_nl
    bulk_nodes_index[i] = bulk_nodes
    bulk_node_input_index[i] = bulk_node_input


avg_acc = sum(acc_max) / len(acc_max)
avg_batch = sum(bulk_b_index) / len(bulk_b_index)
avg_epoch = sum(bulk_e_index) / len(bulk_e_index)
avg_nl = sum(bulk_nl_index) / len(bulk_nl_index)
avg_nodes = sum(bulk_nodes_index) / len(bulk_nodes_index)
avg_node_input = sum(bulk_node_input_index) / len(bulk_node_input_index)

print("Average Accuracy: %.2f" % (avg_acc))
print("Average bulk batch: %.0f" % (avg_batch))
print("Average bulk epoch: %.0f" % (avg_epoch))
print("Average bulk number of layers: %.0f" % (avg_nl))
print("Average bulk number of nodes per layer: %.0f" % (avg_nodes))
print("Average bulk input nodes: %.0f" % (avg_node_input))
