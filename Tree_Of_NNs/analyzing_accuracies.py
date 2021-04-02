import numpy as np

acc = np.genfromtxt('accuracies.tsv', delimiter='\t')
leaf_batch = np.genfromtxt('leaf_batch_size.tsv', delimiter='\t')
leaf_epoch = np.genfromtxt('leaf_epochs.tsv', delimiter='\t')
root_batch = np.genfromtxt('root_batch.tsv', delimiter='\t')
root_epoch = np.genfromtxt('root_epochs.tsv', delimiter='\t')

acc_indeces = np.zeros(20)
acc_max = np.zeros(20)
lf_b_index = np.zeros(20)
lf_e_index = np.zeros(20)
rt_b_index = np.zeros(20)
rt_e_index = np.zeros(20)

new_acc = acc.tolist()
for i in range(len(acc_indeces)):
    max_num = max(new_acc)
    max_index = new_acc.index(max_num)
    acc_indeces[i] = max_index
    acc_max[i] = max_num
    new_acc.remove(max_num)
    lf_b = leaf_batch[max_index]
    lf_e = leaf_epoch[max_index]
    rt_b = root_batch[max_index]
    rt_e = root_epoch[max_index]
    lf_b_index[i] = lf_b
    lf_e_index[i] = lf_e
    rt_b_index[i] = rt_b
    rt_e_index[i] = rt_e


avg_acc = sum(acc_max) / len(acc_max)
avg_lf_batch = sum(lf_b_index) / len(lf_b_index)
avg_lf_epoch = sum(lf_e_index) / len(lf_e_index)
avg_rt_batch = sum(rt_b_index) / len(rt_b_index)
avg_rt_epoch = sum(rt_e_index) / len(rt_e_index)

print("Average Accuracy: %.2f" % (avg_acc))
print("Average leaf batch: %.0f" % (avg_lf_batch))
print("Average leaf epoch: %.0f" % (avg_lf_epoch))
print("Average root batch: %.0f" % (avg_rt_batch))
print("Average root epoch: %.0f" % (avg_rt_epoch))
