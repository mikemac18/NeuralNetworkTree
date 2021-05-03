# NeuralNetworkTree
Testing performance of neural network tree versus a single (bulk) neural network. Our goal was to see if there are any performance benefits in terms of runtime and memory when using a tree of neural networks versus a regular (bulk) neural network. The description of each file and its purpose is below.

Our results are shown in the accuracy_vs_runtime.png file, as well as in the performance folders. We found that on average, the tree had higher accuracy and faster runtime, and also only needed to be trained on half the data as compared to the bulk network. 

Overview of Files:

- Final_Data_Mappings_10000.xlsx -> Our data mappings for the data we used to train and test our neural networks with. We created the feature data by randomly assigning a number within a certain range with a Gaussian distribution. The labels were obtained by mapping each range in the feature data, to a specific "star rating", or label
- Vivek_Ranged_Data.csv -> Our hardcoded data from the Final Mappings Data File for the bulk neural network
- Vivek_Ranged_Tree_Data.csv - Our hardcoded data from the Final Mappings Data File for the tree neural network
- Data_Screenshot.png -> Screenshot of our data
- accuracy_vs_runtime.png -> Plot that compares the accuracy and runtime of the tree versus bulk (regular) neural network, for 20 runs.
- Code_to_Generate_Gaussians.py -> This file was used to generate our random feature data, that was Gaussian distributed.

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


bulk_NN:
- NeuralNetwork.ipynb -> Our jupyter notebook that we used to build our bulk neural network.
- testing_bulk.py -> This file was used to test our neural network, to find the optimal number of hidden layers, number of nodes per layer, batch size, and number of epochs. We ran an infinite loop and initialized random values for these parameters, and recorded their values along with the accuracy and runtime, for each run. These values were outputted to all the different .tsv files.
- analyzing_bulk.py -> This file was used to analyze the data we collected in all of our .tsv files. We wanted to find the run with the highest accuracy, and lowest runtime.
- .tsv files -> These files are where we outputted our values for each parameter after each time the bulk neural network ran.

Bulk_Performance:
- Bulk NN - memory test with mprof.py -> This file was used to test the memory use of the bulk NN.
- Bulk NN - stopwatch test for 20 runs.html -> Log of runtimes and accuracies for 20 runs with the bulk NN with its best parameters.
- Bulk NN - stopwatch test for 20 runs.py -> This file was used to run the bulk NN 20 times with the best parameters, and output the accuracy and runtime for each run.
