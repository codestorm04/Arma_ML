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



## TODO:
1. Build strategies:   [Reference](https://www.cnblogs.com/Anker/p/3527677.html)
2. decision tree pruning
3. models: Bayes, knn, svm, LDA, gbdt, PCA, MDS, k-means, FFM
4. params setting
5. normalization: L1 L2
6. optimazers other than SGD
7. 
8. model saver / loader