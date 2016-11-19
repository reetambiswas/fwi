CC :=nvcc
CCFLAGS :=-Wno-deprecated-gpu-targets -g -G -pg
SRC :=main.cu operator.cpp fileHandling.cpp  mathematicalOperation.cpp global.cpp kernel.cu
EXEC :=forward


$(EXEC) : $(SRC)
	$(CC) $(CCFLAGS) -o $@ $^

.PHONY: clean
clean:
	rm $(EXEC)







# nvcc -Wno-deprecated-gpu-targets  main.cu operator.cpp fileHandling.cpp  mathematicalOperation.cpp global.cpp kernel.cu

#./a.out
