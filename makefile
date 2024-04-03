
ProjectRoot=.

include config.mk

include $(ClPlatform)/ClPlatform.mk

all: matrix_operations_tests

clean: ClPlatformClean
	@rm -f lib/*.a obj/* nbodyProject

obj/%.o: source/%.cpp
	@echo "\tCXX\t"$*.o
	@g++ -fopenmp $^ -o $@ $(AllInclude) $(cpp_flags) -O2

matrix_operations_tests_obj += obj/main.o
matrix_operations_tests_obj += obj/Matrix.o
matrix_operations_tests_obj += obj/MultiMatrix.o
matrix_operations_tests_obj += obj/MultiMatrixT.o
matrix_operations_tests_obj += obj/MatrixSwapped.o
matrix_operations_tests_obj += obj/time.o
matrix_operations_tests_obj += obj/MatrixMultiplyTest.o
matrix_operations_tests_obj += obj/MultiMatrixMultiplyTest.o
matrix_operations_tests_obj += obj/testLists.o
matrix_operations_tests_obj += obj/NeuronLayer.o
matrix_operations_tests_obj += obj/NeuronLayerTest.o

matrix_operations_tests: $(matrix_operations_tests_obj) $(libClPlatform)
	@echo "\tLD\matrix_operations_tests"
	@g++ -o $@ $^ $(OpenClLib) $(OpenGL) -fopenmp -O2
