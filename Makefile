CC = g++
PROJECT = libarma_ml.so

INCLUDES = $(shell find ./include -name "*.h")
SRC = $(shell find ./src -name "*.cc")
OBJ = $(SRC:%.cc=%.o) 
CXXFLAGS = -std=c++14 -larmadillo -I ./include/ -fPIC
LINKFLAGS = -larmadillo -shared -fPIC
PREFIX = /usr



$(PROJECT): $(OBJ)
	$(CC) -o $(PROJECT) $(OBJ) $(LINKFLAGS)
	@echo "[Successfully Build.]"

%.o: %.c $(INCLUDES)
	$(CC) -c $< -o $@  $(CXXFLAGS)

.PHONY: install
install: $(PROJECT)
	install $(PROJECT) $(PREFIX)/lib/
	mkdir $(PREFIX)/include/arma_ml
	cp ./include/* $(PREFIX)/include/arma_ml/ -r
	@echo export CPLUS_INCLUDE_PATH=$PLUS_INCLUDE_PATH:$(PREFIX)/include/arma_ml/ >> ~/.bashrc
	source ~/.bashrc
	@echo "[Successfully installed.]"


.PHONY: clean
clean:
	rm -rf $(OBJ) $(PROJECT)
	@echo "[Successfully Cleaned.]"