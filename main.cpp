// 0.3v No Stable
#include "Headers/NeuralNetWork.h"

int main()
{
	//std::vector<std::vector<double>> inputSet = { {0, 0}, {0, 1}, {1, 0}, {1, 1} };
	//std::vector<std::vector<double>> outputSet = { {0}, {1}, {1}, {0} };
	std::vector<std::string> planets =
	{
		"Sun",
		"Mercury",
		"Venus",
		"Earth",
		"Mars",
		"Jupiter"
	};
	std::vector<std::vector<double>> inputSet =
	{
		{696340.0f / 10000.0},
		{2439.7 / 10000.0},
		{6051.8 / 10000.0},
		{7300.0f / 10000.0},
		{3389.5 / 10000.0},
		{69911 / 10000.0}
	};
	std::vector<std::vector<double>> outputSet = 
	{ 
		{1, 0, 0, 0, 0, 0},
		{0, 1, 0, 0, 0, 0},
		{0, 0, 1, 0, 0, 0}, 
		{0, 0, 0, 1, 0, 0},
		{0, 0, 0, 0, 1, 0},
		{0, 0, 0, 0, 0, 1}
	};
	//NeuralNetwork nn(1, { 10 }, 6, 0.1, 0, true);
	NeuralNetwork nn("nn.txt");
	//nn.trainToIterarion(inputSet, outputSet, 10000, true);

	//nn.printResultTrain(inputSet, outputSet);
	nn.predict({ 0 / 10000.0 });
	for (int i = 0; i < planets.size(); i++) 
	{
		if (std::round(nn.getLayers()[nn.getLayers().size() - 1][i].getValue()) == 1) 
		{
			std::cout << planets[i];
		}
	}
	//matplot::show();
	return 0;
}