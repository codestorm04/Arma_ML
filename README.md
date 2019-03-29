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
4. numpy darrays testing
5. debugging with armadillo in gdb