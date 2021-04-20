# NeuralNetworkTree
Testing performance of neural network tree versus a single (bulk) neural network

Overview of Files:

- Final_Data_Mappings_10000.xlsx -> Our data mappings for the data we used to train and test our neural networks with. We created the feature data by randomly assigning a number within a certain range with a Gaussian distribution. The labels were obtained by mapping each range in the feature data, to a specific "star rating", or label
- Vivek_Ranged_Data.csv (This file is in all folders) -> Our hardcoded data from the Final Mappings Data File
- Data_Screenshot.png -> Screenshot of our data
- accuracy_vs_runtime.png -> Plot that compares the accuracy and runtime of the tree versus bulk (regular) neural network, for 20 runs.

bulk_NN:
- NeuralNetwork.ipynb -> Our jupyter notebook that we used to build our bulk neural network.
- testing_bulk.py -> This file was used to test our neural network, to find the optimal number of hidden layers, number of nodes per layer, batch size, and number of epochs. We ran an infinite loop and initialized random values for these parameters, and recorded their values along with the accuracy and runtime, for each run. These values were outputted to all the different .tsv files.
- analyzing_bulk.py -> This file was used to analyze the data we collected in all of our .tsv files. We wanted to find the run with the highest accuracy, and lowest runtime.
- .tsv files -> These files are where we outputted our values for each parameter after each time the bulk neural network ran.

Tree_Of_NNs:
- NN_Tree.ipynb -> Our jupyter notebook that we used to build our tree neural network.

  Relu (Everything used in this folder was with rectified linear activation function):
  - testing_tree_relu.py -> This file was used to test our tree neural network to find the optimal number of batch size and epochs for the leaf networks and root network. We ran an infinite loop and initialized random values for these parameters, and recorded their values along with the accuracy and runtime for each run. These values were outputted to all the different .tsv files.
  - analyzing_tree_relu.py -> This file was used to analyze the data we collected in all of our .tsv files. We wanted to find the best runs, with highest accuracy and lowest runtime.
  - .tsv files -> These files are where we outputted our values for each parameter after each time the tree neural network ran.

  Sigmoid (Everything used in this folder was with the sigmoid activation function):
  - Everything is the same as Relu folder

Tree_Performance:
- testing_performance_tree_relu.py -> Using the best parameters found in the analyzing_tree_relu.py file, we run the tree 20 times and record the runtimes and accuracies for each run.
- tree_memory_testing.py -> Using the best parameters found in the analyzing_tree.py files to get an mprof memory plot for the tree for both the sigmoid and rectified linear activation functions.
- runtime_vs_accuracy_perf.xlsx -> This excel file was used to create our plot comparing the accuracies and runtimes of the bulk versus tree neural networks for 20 runs, with their best parameters.
- .tsv files -> Where we outputted the runtimes and accuracies of the 20 runs with the best parameters.
- mprof .png files -> Our memory plots for the tree neural network with the sigmoid and rectified linear activation functions
- runtime_vs_accuracy plots -> Plots produced from the excel file.


modification @ 18:36
