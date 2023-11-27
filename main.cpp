// 0.3v No Stable
#include "Headers/NeuralNetWork.h"

int main()
{
	std::vector<std::vector<double>> inputSet = { {0, 0}, {0, 1}, {1, 0}, {1, 1} };

	std::vector<double> outputSet = { 0, 1, 1, 1 };

	//NeuralNetwork nn(2, { 2 }, 1, 1, 0.8, false);
	NeuralNetwork nn("nn.txt");
	//nn.trainToIterarion(inputSet, outputSet, 10000, true);

	nn.printResultTrain(inputSet);
	//matplot::show();
	return 0;
}