#include "../Headers/NeuralNetwork.h"

double NeuralNetwork::activate(double x)
{
	return 1.0 / (1.0 + exp(-x));
}
double NeuralNetwork::derivative(double x)
{
	return x * (1 - x);
}
NeuralNetwork::NeuralNetwork(int countInputNeuron, std::vector<int> countHideNeuron, int countOutputNeuron, double learningRate, double momentNes, bool useBias)
{
	this->learningRate = learningRate;
	this->momentNes = momentNes;
	this->useBias = useBias;
	this->layers.resize(countHideNeuron.size() + 2);
	//this->allLayers.resize(2);
	// Генерация выходных нейронов
	for (int i = 0; i < countOutputNeuron; i++)
	{
		layers[layers.size() - 1].push_back(Neuron());
	}
	// Генерация скрытых слоев и скрытых нейронов
	for (int i = countHideNeuron.size(); i > 0; i--)
	{
		if (countHideNeuron[i - 1] == 0) continue;
		//allLayers.push_back(std::vector<Neuron>());
		for (int j = 0; j < countHideNeuron[i - 1]; j++)
		{
			layers[i].push_back(Neuron(layers[i + 1].size()));
		}
		if (useBias)
		{
			layers[i].push_back(Neuron(layers[i + 1].size(), 1, true));
		}
	}
	// Генерация входных нейронов
	for (int i = 0; i < countInputNeuron; i++)
	{
		layers[0].push_back(Neuron(layers[1].size()));
	}
	if (useBias) layers[0].push_back(Neuron(layers[1].size(), 1, true));
}
double NeuralNetwork::getResult()
{
	return layers[layers.size() - 1][0].getValue();
}
double NeuralNetwork::getErrorSquare()
{
	return pow(getError(), 2);
}
double NeuralNetwork::getError()
{
	return layers[layers.size() - 1][0].getError();
}
void NeuralNetwork::forwardPropagation(std::vector<double> inputs)
{
	// Установка входных параметров
	for (int i = 0; i < layers[0].size() - 1; i++)
	{
		layers[0][i].setValue(inputs[i]);
	}
	// Поиск сумм для остальных слоев (включая выходной)
	for (int i = 1; i < layers.size(); i++)
	{
		for (int j = 0; j < layers[i].size(); j++)
		{
			if (layers[i][j].getBias())
			{
				layers[i][j].setValue(1);
				continue;
			}
			double sum = 0.0;
			for (int k = 0; k < layers[i - 1].size(); k++)
			{
				sum += layers[i - 1][k].getValue() * layers[i - 1][k].getWeights()[j];
			}
			layers[i][j].setValue(activate(sum));
		}
	}
}
void NeuralNetwork::backPropagation(std::vector<double> inputs, double value)
{
	forwardPropagation(inputs);
	// Рассчет ошибки и градиента выходных нейронов
	for (int i = 0; i < layers[layers.size() - 1].size(); i++)
	{
		layers[layers.size() - 1][i].setError(value - layers[layers.size() - 1][i].getValue());
		layers[layers.size() - 1][i].setGradient(layers[layers.size() - 1][i].getError() * derivative(layers[layers.size() - 1][i].getValue()));
	}

	// ошибки для скрытых слоев
	for (int i = layers.size() - 2; i > 0; i--)
	{
		for (int j = 0; j < layers[i].size(); j++)
		{
			double sum = 0.0;
			for (int k = 0; k < layers[i + 1].size(); k++)
			{
				sum += layers[i + 1][k].getGradient() * layers[i][j].getWeights()[k];
			}
			layers[i][j].setGradient(sum * derivative(layers[i][j].getValue()));
		}
	}
	// Корректирование весов для всех слоев
	for (int i = 0; i < layers.size() - 1; i++)
	{
		for (int j = 0; j < layers[i].size(); j++)
		{
			for (int k = 0; k < layers[i + 1].size(); k++)
			{
				double deltaW = learningRate * layers[i + 1][k].getGradient() * layers[i][j].getValue() + momentNes * layers[i][j].getLastDeltaWeights()[k];
				layers[i][j].setWeights(k, layers[i][j].getWeights()[k] + deltaW);
				layers[i][j].setLastDeltaWeights(k, deltaW);
			}
		}
	}
}
void NeuralNetwork::trainToIterarion(std::vector<std::vector<double>>& inputSet, std::vector<double>& outputSet, int iteration)
{
	for (int i = 0; i < iteration; i++)
	{
		double sum = 0;
		std::vector<double> errors;
		for (int j = 0; j < inputSet.size(); j++)
		{
			backPropagation(inputSet[j], outputSet[j]);
			sum += getErrorSquare();
		}
		coordinatesX.push_back(i);
		coordinatesY.push_back(sum / inputSet.size());
	}
	matplot::plot(coordinatesX, coordinatesY);
}
void NeuralNetwork::trainBeforeTheError(std::vector<std::vector<double>>& inputSet, std::vector<double>& outputSet, double errorMax, int maxIteration)
{
	double error = (double)INFINITE;
	int i = 0;
	while (error > errorMax)
	{
		double sum = 0;
		std::vector<double> errors;
		for (int j = 0; j < inputSet.size(); j++)
		{
			backPropagation(inputSet[j], outputSet[j]);
			sum += getErrorSquare();
		}
		coordinatesX.push_back(i);
		coordinatesY.push_back(sum / inputSet.size());
		error = sum / inputSet.size();
		if (i > maxIteration)
		{
			std::cout << "Over iteration\n";
			break;
		}
		i++;
	}
	std::cout << "Iteration " << i << "\n";
	matplot::plot(coordinatesX, coordinatesY);
}
void NeuralNetwork::printResultTrain(std::vector<std::vector<double>>& inputSet)
{
	for (int i = 0; i < inputSet.size(); i++)
	{
		forwardPropagation(inputSet[i]);
		std::cout << i + 1 << " " << getResult() << "\n";
		layers.end();
	}
	//for (int i = 0; i < layers[layers.size() - 1].size(); i++)
	//{
	//	std::cout << i + 1 << " " << layers[layers.size() - 1][i].getValue() << "\n";
	//}
}
void NeuralNetwork::saveWeights()
{
	std::ofstream file("nn.txt");
	for (int i = 0; i < layers.size(); i++) 
	{
		file << i + 1 << " слой\n";
		for (int j = 0; j < layers[i].size(); j++) 
		{
			file << j + 1 << " нейрон\n";
			for (int k = 0; k < layers[i][j].getWeights().size(); k++)
			{
				file << layers[i][j].getWeights()[k] << std::endl;
			}
		}
	}
	file.close();
	std::cout << "save";
}