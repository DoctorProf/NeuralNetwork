// 0.21v No Stable
#include <iostream>
#include <vector>
#include <cmath>
#include <stdlib.h>
#include <time.h>
#include "matplot/matplot.h"

class Neuron 
{
private:
	std::vector<double> weights;
	std::vector<double> lastDeltaWeights;
	bool toBias;
	double value;
	double error;
	double gradient;

	void initializeWeights() 
	{
		srand(time(NULL));
		for (size_t i = 0; i < weights.size(); i++) 
		{
			weights[i] = (rand() % 100) / 50.0 - 1;
			//weights[i] = 0.5;
		}
	}
	void initializeDeltaWeights()
	{
		for (size_t i = 0; i < lastDeltaWeights.size(); i++)
		{
			lastDeltaWeights[i] = 0;
		}
	}
public:
	Neuron(int countWeight = 0, double value = 0, bool toBias = false) 
	{
		this->value = value;
		this->weights.resize(countWeight);
		this->lastDeltaWeights.resize(countWeight);
		this->toBias = toBias;
		initializeWeights();
		initializeDeltaWeights();
	}
	void setValue(double value)
	{
		this->value = value;
	}
	double getValue()
	{
		return this->value;
	}
	void setError(double error)
	{
		this->error = error;
	}
	double getError()
	{
		return this->error;
	}
	void setGradient(double gradient)
	{
		this->gradient = gradient;
	}
	double getGradient()
	{
		return this->gradient;
	}
	void setWeights(int index, double value)
	{
		this->weights[index] = value;
	}
	std::vector<double> getWeights()
	{
		return this->weights;
	}
	void setLastDeltaWeights(int index, double value)
	{
		this->lastDeltaWeights[index] = value;
	}
	std::vector<double> getLastDeltaWeights()
	{
		return this->lastDeltaWeights;
	}
	bool getBias()
	{
		return this->toBias;
	}
};
class NeuralNetwork 
{
private:
	bool useBias;
	double momentNes;
	double learningRate;
	std::vector<std::vector<Neuron>> allLayers;
	std::vector<int> coordinatesX;
	std::vector<double> coordinatesY;

	double activate(double x)
	{
		return 1.0 / (1.0 + exp(-x));
	}
	double derivative(double x)
	{
		return x * (1 - x);
	}
public:
	NeuralNetwork(int countInputNeuron, std::vector<int> countHideNeuron, int countoutputNeuron, double learningRate, double momentNes, bool useBias)
	{
		this->learningRate = learningRate;
		this->momentNes = momentNes;
		this->useBias = useBias;
		this->allLayers.resize(countHideNeuron.size() + 2);
		//this->allLayers.resize(2);
		for (int i = 0; i < countoutputNeuron; i++)
		{
			allLayers[allLayers.size() - 1].push_back(Neuron());
		}
		
		for (int i = countHideNeuron.size(); i > 0; i--)
		{
			if (countHideNeuron[i - 1] == 0) continue;
			//allLayers.push_back(std::vector<Neuron>());
			for (int j = 0; j < countHideNeuron[i - 1]; j++)
			{
				if (i == countHideNeuron.size()) 
				{
					allLayers[i].push_back(Neuron(allLayers[allLayers.size() - 1].size()));
				}
				else 
				{
					allLayers[i].push_back(Neuron(allLayers[i + 1].size() + useBias));
				}
			}
			if (useBias) 
			{
				if (i == countHideNeuron.size())
				{
					allLayers[i].push_back(Neuron(allLayers[allLayers.size() - 1].size(), 1, true));
				}
				else
				{
					allLayers[i].push_back(Neuron(allLayers[i + 1].size() + useBias, 1, true));
				}

			}
		}
		for (int i = 0; i < countInputNeuron; i++)
		{
			allLayers[0].push_back(Neuron(allLayers[1].size()));
		}
		if (useBias) allLayers[0].push_back(Neuron(allLayers[1].size(), 1, true));
	}
	double getResult()
	{
		return allLayers[allLayers.size() - 1][0].getValue();
	}
	double getErrorSquare()
	{
		return pow(getError(), 2);
	}
	double getError()
	{
		return allLayers[allLayers.size() - 1][0].getError();
	}
	void forwardPropagation(std::vector<double> inputs)
	{
		for (int i = 0; i < allLayers[0].size() - 1; i++)
		{
			allLayers[0][i].setValue(inputs[i]);
		}
		for (int i = 1; i < allLayers.size(); i++)
		{
			for (int j = 0; j < allLayers[i].size(); j++)
			{
				if (allLayers[i][j].getBias())
				{
					allLayers[i][j].setValue(1);
					continue;
				}
				double sum = 0.0;
				for (int k = 0; k < allLayers[i - 1].size(); k++) 
				{
					sum += allLayers[i - 1][k].getValue() * allLayers[i - 1][k].getWeights()[j];
				}
				allLayers[i][j].setValue(activate(sum));
			}
		}
	}
	void backPropagation(std::vector<double> inputs, int value)
	{
		forwardPropagation(inputs);
		for (int i = 0; i < allLayers[allLayers.size() - 1].size(); i++)
		{
			allLayers[allLayers.size() - 1][i].setError(value - allLayers[allLayers.size() - 1][i].getValue());
			allLayers[allLayers.size() - 1][i].setGradient(allLayers[allLayers.size() - 1][i].getError() * derivative(allLayers[allLayers.size() - 1][i].getValue()));
		}
		for (int i = allLayers.size() - 2; i > 0; i--)
		{
			for (int j = 0; j < allLayers[i].size(); j++)
			{
				
				double sum = 0.0;
				for (int k = 0; k < allLayers[i + 1].size(); k++)
				{
					sum += allLayers[i + 1][k].getGradient() * allLayers[i][j].getWeights()[k];
				}
				allLayers[i][j].setError(sum);
			}
		}
		for (int i = allLayers.size() - 2; i > 0; i--)
		{
			for (int j = 0; j < allLayers[i].size(); j++)
			{
				allLayers[i][j].setGradient(allLayers[i][j].getError() * derivative(allLayers[i][j].getValue()));
			}
		}
		for (int i = allLayers.size() - 2; i > -1 ; i--)
		{
			for (int j = 0; j < allLayers[i].size(); j++)
			{
				for (int k = 0; k < allLayers[i + 1].size(); k++)
				{
					double deltaW = learningRate * allLayers[i + 1][k].getGradient() * allLayers[i][j].getValue() + momentNes * allLayers[i][j].getLastDeltaWeights()[k];
					allLayers[i][j].setWeights(k, allLayers[i][j].getWeights()[k] + deltaW);
					allLayers[i][j].setLastDeltaWeights(k, deltaW);
				}
			}
		}
	}
	void trainToIterarion(std::vector<std::vector<double>> inputSet, std::vector<double> outputSet, int iteration)
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
			coordinatesY.push_back((1.0f / inputSet.size()) * sum);
		}
		//matplot::plot(coordinatesX, coordinatesY);
	}
	void trainBeforeTheError(std::vector<std::vector<double>> inputSet, std::vector<double> outputSet, double errorMax, int maxIteration)
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
			coordinatesY.push_back((1.0f / inputSet.size()) * sum);
			error = (1.0f / inputSet.size()) * sum;
			if (i > maxIteration)
			{
				break;
			}
			i++;
		}
		//matplot::plot(coordinatesX, coordinatesY);
	}
	void printResultTrain(std::vector<std::vector<double>> inputSet)
	{
		for (int i = 0; i < inputSet.size(); i++)
		{
			forwardPropagation(inputSet[i]);
			std::cout << i + 1 << " " << getResult() << "\n";
		}
	}
	void saveWeights()
	{

	}
};
int main()
{
	std::vector<std::vector<double>> inputSet = { {0, 0}, {0, 1}, {1, 0}, {1, 1} };
	//std::vector<std::vector<double>> inputSet = { {0}, {1} };
	std::vector<double> outputSet = { 0, 1, 1, 0 };
	NeuralNetwork nn(2, { 2, 2 }, 1, 1, 0.8, true);
	//nn.printResultTrain(inputSet);
	clock_t start = clock();
	//nn.trainBeforeTheError(inputSet, outputSet, 0.05, 10000);
	nn.trainToIterarion(inputSet, outputSet, 1000);
	clock_t end = clock();
	std::cout << (end - start) / (CLOCKS_PER_SEC / 1000.0f) << "\n";
	nn.printResultTrain(inputSet);
	//matplot::show();
	return 0;
}