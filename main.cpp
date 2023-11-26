// 0.21v No Stable
#include "Headers/NeuralNetWork.h"
#include <time.h>


int main()
{
	std::vector<std::vector<double>> inputSet = { {0, 0}, {0, 1}, {1, 0}, {1, 1} };
	//std::vector<std::vector<double>> inputSet = { {0}, {1} };

	std::vector<double> outputSet = { 0, 1, 1, 1 };

	NeuralNetwork nn(2, { 2 }, 1, 2, 0.80, true);

	clock_t start = clock();

	nn.trainToIterarion(inputSet, outputSet, 1000);

	clock_t end = clock();

	std::cout << (end - start) / (CLOCKS_PER_SEC / 1000.0f) << "\n";

	nn.printResultTrain(inputSet);

	//matplot::show();
	return 0;
}