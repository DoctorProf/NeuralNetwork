// 0.21v No Stable
#include "Headers/NeuralNetWork.h"
#include <time.h>


int main()
{
	std::vector<std::vector<double>> inputSet = { {0, 0}, {0, 1}, {1, 0}, {1, 1} };
	//std::vector<std::vector<double>> inputSet = { {0}, {1} };

	std::vector<double> outputSet = { 0, 1, 1, 1 };

	NeuralNetwork nn(2, { 2 }, 1, 1, 0.8, true);

	clock_t start = clock();

	//nn.trainToIterarion(inputSet, outputSet, 10000);
	//nn.trainBeforeTheError(inputSet, outputSet, 0.01, 1000000);
	clock_t end = clock();

	std::cout << (end - start) / (CLOCKS_PER_SEC / 1000.0f) << "\n";
	nn.printResultTrain(inputSet);
	nn.saveWeights();
	matplot::show();
	return 0;
}