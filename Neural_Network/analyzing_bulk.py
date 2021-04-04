import numpy as np

bulk_acc = np.genfromtxt('bulk_accuracies.tsv', delimiter='\t')
bulk_batch = np.genfromtxt('bulk_batch.tsv', delimiter='\t')
bulk_epoch = np.genfromtxt('bulk_epochs.tsv', delimiter='\t')

acc_indeces = np.zeros(20)
acc_max = np.zeros(20)
bulk_b_index = np.zeros(20)
bulk_e_index = np.zeros(20)
new_acc = bulk_acc.tolist()

for i in range(len(acc_indeces)):
    max_num = max(new_acc)
    max_index = new_acc.index(max_num)
    acc_indeces[i] = max_index
    acc_max[i] = max_num
    new_acc.remove(max_num)
    bulk_b = bulk_batch[max_index]
    bulk_e = bulk_epoch[max_index]
    bulk_b_index[i] = bulk_b
    bulk_e_index[i] = bulk_e


avg_acc = sum(acc_max) / len(acc_max)
avg_bulk_batch = sum(bulk_b_index) / len(bulk_b_index)
avg_bulk_epoch = sum(bulk_e_index) / len(bulk_e_index)

print("Average Accuracy: %.2f" % (avg_acc))
print("Average bulk batch: %.0f" % (avg_bulk_batch))
print("Average bulk epoch: %.0f" % (avg_bulk_epoch))
