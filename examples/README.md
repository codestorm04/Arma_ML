# To run the examples, compile the xx_test.cc w.r.t. different moduels.
```c++
g++ xx_test.cc -std=c++14 -larmadillo -L.. -larma_ml -Wl,--rpath=.. -I ../include/
```