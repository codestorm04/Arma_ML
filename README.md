# Machine_Learning_CPP
Author: Yuzhen Liu  
Started From 2019.3.22 (Two days after my birthday)  
Lightweighted Statistic ML implementations on C++.  


## Installation
Requeirs C++ algebra library armadillo, installation of armadillo is as follows:  

Tested for Ubuntu 16.04:  
    
    sudo apt-get install libopenblas-dev
	sudo apt-get install liblapack-dev
	sudo apt-get install libarpack2-dev
	sudo apt-get install libsuperlu-dev

download armadillo .tar, and build it.  

	cmake .
	make
	sudo make install



TODO:
1. params setting
2. normalization: L1 L2
3. optimazations other than SGD
4. models: linear regression, knn, svm, LDA, decision tree, gbdt, random forest, PCA, MDS, k-means, FM
5. model saver / loader
6. Build strategies:   [Reference](https://www.cnblogs.com/Anker/p/3527677.html)
```c++
cd src
g++ -shared -fPIC datasets/datasets.cc -std=c++14 -o libdata.so -I ../include/ -larmadillo
g++ -shared -fPIC logistic/softmax_classifier.cc -std=c++14 -o libsoftmaxclassifier.so -I ../include/ -larmadillo
g++ test.cc -I ../../include/  -L ./ -larmadillo -ldata   -std=c++14 -lsoftmaxclassifier
export LD_LIBRARY_PATH=/home/lyz/desktop/github_repos/Machine_Learning_CPP/src/logisitc
```
