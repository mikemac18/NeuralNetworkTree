import numpy as np

acc = np.genfromtxt('accuracies_relu.tsv', delimiter='\t')
leaf_batch = np.genfromtxt('leaf_batch_size_relu.tsv', delimiter='\t')
leaf_epoch = np.genfromtxt('leaf_epochs_relu.tsv', delimiter='\t')
root_batch = np.genfromtxt('root_batch_relu.tsv', delimiter='\t')
root_epoch = np.genfromtxt('root_epochs_relu.tsv', delimiter='\t')
runtime = np.genfromtxt('tree_runtimes_relu.tsv', delimiter='\t')

print(sum(acc) / len(acc))

acc_indeces = np.zeros(5)
acc_max = np.zeros(5)
lf_b_index = np.zeros(5)
lf_e_index = np.zeros(5)
rt_b_index = np.zeros(5)
rt_e_index = np.zeros(5)
runtime_index = np.zeros(5)

new_acc = acc.tolist()
'''for i in range(len(acc_indeces)):
    max_num = max(new_acc)
    max_index = new_acc.index(max_num)
    acc_indeces[i] = max_index
    acc_max[i] = max_num
    new_acc.remove(max_num)
    lf_b = leaf_batch[max_index]
    lf_e = leaf_epoch[max_index]
    rt_b = root_batch[max_index]
    rt_e = root_epoch[max_index]
    runtime_max = runtime[max_index]
    lf_b_index[i] = lf_b
    lf_e_index[i] = lf_e
    rt_b_index[i] = rt_b
    rt_e_index[i] = rt_e
    runtime_index[i] = runtime_max'''

#runtime_temp = runtime[1]
#acc_temp = acc[1]
leaf_b = []
leaf_e = []
root_b = []
root_e = []
rtime = []
accs = []
for i in range(len(runtime)):
    #if(runtime[i] < runtime_temp and acc[i] > acc_temp):
    if(runtime[i] < 70 and acc[i] > 85):
        leaf_b.append(leaf_batch[i])
        leaf_e.append(leaf_epoch[i])
        root_b.append(root_batch[i])
        root_e.append(root_epoch[i])
        rtime.append(runtime[i])
        accs.append(acc[i])
        #runtime_temp = runtime[i]
        #acc_temp = acc[i]

print(sum(leaf_b) / len(leaf_b))
print(sum(leaf_e) / len(leaf_e))
print(sum(root_b) / len(root_b))
print(sum(root_e) / len(root_e))
print(sum(rtime) / len(rtime))
print(sum(accs) / len(accs))
print(leaf_b)
print(leaf_e)
print(root_b)
print(root_e)
print(rtime)
print(accs)
#print(acc_temp)
#print(runtime_temp)
#avg_acc = sum(acc_max) / len(acc_max)
#avg_lf_batch = sum(lf_b_index) / len(lf_b_index)
#avg_lf_epoch = sum(lf_e_index) / len(lf_e_index)
#avg_rt_batch = sum(rt_b_index) / len(rt_b_index)
#avg_rt_epoch = sum(rt_e_index) / len(rt_e_index)
#avg_runtime = sum(runtime_index) / len(runtime_index)

#print("Average Accuracy of top five: %.2f" % (avg_acc))
#print("Average leaf batch of top five: %.0f" % (avg_lf_batch))
#print("Average leaf epoch of top five: %.0f" % (avg_lf_epoch))
#print("Average root batch of top five: %.0f" % (avg_rt_batch))
#print("Average root epoch of top five: %.0f" % (avg_rt_epoch))
#print("Average runtime of top five: %f" % (avg_runtime))
#print("Top five accuracies:")
#print(acc_max)
