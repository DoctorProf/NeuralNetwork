// 0.3v No Stable
#include "Headers/NeuralNetWork.h"

double normalize(double value, double minValue, double maxValue) 
{
	double x = (value - minValue) / (maxValue - minValue);
	return (value - minValue) / (maxValue - minValue);
}
int main()
{
	std::setlocale(LC_ALL, "rus");
	//std::vector<std::vector<double>> inputSet = { {0, 0}, {0, 1}, {1, 0}, {1, 1} };
	//std::vector<std::vector<double>> outputSet = { {0}, {1}, {1}, {0} };
	std::vector<std::vector<double>> inputSet =
	{
		{0, 0},
		{0, 1},
		{1, 0},
		{1, 1},
	};
	std::vector<std::vector<double>> outputSet = 
	{ 
		{0},
		{1},
		{1},
		{0},
	};
	//NeuralNetwork nn(2, { 2 }, 1, 1, 0, true);
	NeuralNetwork nn("nn.txt");
	nn.trainToIterarion(inputSet, outputSet, 10000, true);
	nn.printResultTrain(inputSet, outputSet);
	/*float radius;
	std::cout << "Введите радиус для предсказания - ";
	std::cin >> radius;
	std::cout << "\n";
	nn.predict({ normalize(radius, 2439.7, 696340.0f) });
	std::vector<Neuron> layers = nn.getLayers()[nn.getLayers().size() - 1];
	std::vector<double> values;
	for (int i = 0; i < layers.size(); i++)
	{
		values.push_back(layers[i].getValue());
	}
	for (int i = 0; i < planets.size(); i++) 
	{
		if (layers[i].getValue() == *max_element(values.begin(), values.end()) && std::round(layers[i].getValue()) == 1)
		{

		}
	}*/
	//matplot::show();
	return 0;
}