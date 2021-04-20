## Importing packages
import numpy as np
import pandas as pd

## Getting data
dataframe = pd.read_csv("bulk_nodes.tsv", delimiter=",", header=None)
bulk_nodes = dataframe.values

## Getting data from tsv files
bulk_acc = np.genfromtxt('bulkNN_accuracies.tsv', delimiter='\t')
bulk_batch = np.genfromtxt('num_batch.tsv', delimiter='\t')
bulk_epoch = np.genfromtxt('num_epochs.tsv', delimiter='\t')
bulk_num_layers = np.genfromtxt('bulk_layers.tsv', delimiter='\t')
#bulk_nodes = np.genfromtxt('bulk_nodes.tsv', delimiter='\t')
input_nodes = np.genfromtxt('input_nodes.tsv', delimiter='\t')
runtimes = np.genfromtxt('bulk_runtimes.tsv', delimiter='\t')

## Printing average accuracy from the bulk accuracies
print(sum(bulk_acc) / len(bulk_acc))

'''
acc_indeces = np.zeros(3)
acc_max = np.zeros(3)
bulk_b_index = np.zeros(3)
bulk_e_index = np.zeros(3)
bulk_nl_index = np.zeros(3)
bulk_nodes_index = np.zeros(3)
bulk_node_input_index = np.zeros(3)
bulk_runtimes = np.zeros(3)
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
    bulk_nodess = bulk_nodes[max_index]
    bulk_node_input = input_nodes[max_index]
    bulk_runtime = runtimes[max_index]
    bulk_b_index[i] = bulk_b
    bulk_e_index[i] = bulk_e
    bulk_nl_index[i] = bulk_nl
    bulk_nodes_index[i] = bulk_nodess
    bulk_node_input_index[i] = bulk_node_input
    bulk_runtimes[i] = bulk_runtime'''

## Getting the data that produced the best results
bulk_b = []
bulk_e = []
bulk_numl = []
bulk_n = []
bulk_in = []
rtime = []
accs = []
for i in range(len(runtimes)):
    #if(runtime[i] < runtime_temp and acc[i] > acc_temp):
    if(runtimes[i] < 500 and bulk_acc[i] > 70):
        bulk_b.append(bulk_batch[i])
        bulk_e.append(bulk_epoch[i])
        bulk_numl.append(bulk_num_layers[i])
        bulk_n.append(bulk_nodes[i])
        bulk_in.append(input_nodes[i])
        rtime.append(runtimes[i])
        accs.append(bulk_acc[i])
        #runtime_temp = runtime[i]
        #acc_temp = acc[i]

## Printing the average best results
print(sum(bulk_b) / len(bulk_b))
print(sum(bulk_e) / len(bulk_e))
print(sum(bulk_numl) / len(bulk_numl))
#print(sum(bulk_n) / len(bulk_n))
print(sum(bulk_in) / len(bulk_in))
print(sum(rtime) / len(rtime))
print(sum(accs) / len(accs))

## Printing best results
print(bulk_b)
print(bulk_e)
print(bulk_numl)
print(bulk_n)
print(bulk_in)
print(rtime)
print(accs)


'''avg_acc = sum(acc_max) / len(acc_max)
avg_batch = sum(bulk_b_index) / len(bulk_b_index)
avg_epoch = sum(bulk_e_index) / len(bulk_e_index)
avg_nl = sum(bulk_nl_index) / len(bulk_nl_index)
#bulk_nodes_index[i][k]
avg_nodes = sum(bulk_nodes_index) / len(bulk_nodes_index)
avg_node_input = sum(bulk_node_input_index) / len(bulk_node_input_index)
avg_runtime = sum(bulk_runtimes) / len(bulk_runtimes)

print("Average Accuracy: %.2f" % (avg_acc))
print("Average bulk batch: %.0f" % (avg_batch))
print("Average bulk epoch: %.0f" % (avg_epoch))
print("Average bulk number of layers: %.0f" % (avg_nl))
print("Average bulk number of nodes per layer: %.0f" % (avg_nodes))
print("Average bulk input nodes: %.0f" % (avg_node_input))
print("Average runtime: %f" % (avg_runtime))'''
