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
	std::vector<double> getsetLastDeltaWeights()
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
	std::vector<Neuron> inputNeurons;
	std::vector<std::vector<Neuron>> hideNeurons;
	std::vector<Neuron> outputNeurons;
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
		this->hideNeurons.resize(countHideNeuron.size());
		for (int i = 0; i < countInputNeuron; i++) 
		{
			if (countHideNeuron.size() == 0) 
			{
				inputNeurons.push_back(Neuron(countoutputNeuron));
			}
			else 
			{
				inputNeurons.push_back(Neuron(countHideNeuron[0] + useBias));
			}
		}
		if (useBias) inputNeurons.push_back(Neuron(countHideNeuron[0] + useBias, 1, true));
		for (int i = 0; i < countHideNeuron.size(); i++)
		{
			for (int j = 0; j < countHideNeuron[i]; j++)
			{
				if (i == countHideNeuron.size() - 1) 
				{
					hideNeurons[i].push_back(Neuron(countoutputNeuron));
				}
				else 
				{
					hideNeurons[i].push_back(Neuron(countHideNeuron[i + 1] + useBias));
				}
			}
			if (useBias) 
			{
				if (i == countHideNeuron.size() - 1)
				{
					hideNeurons[i].push_back(Neuron(countoutputNeuron, 1, true));
				}
				else
				{
					hideNeurons[i].push_back(Neuron(countHideNeuron[i + 1] + useBias, 1, true));
				}
			}
		}
		for (int i = 0; i < countoutputNeuron; i++)
		{
			outputNeurons.push_back(Neuron());
		}
	}
	double getResult()
	{
		return outputNeurons[0].getValue();
	}
	double getErrorSquare()
	{
		return pow(outputNeurons[0].getError(), 2);
	}
	double getError()
	{
		return outputNeurons[0].getError();
	}
	void forwardPropagation(std::vector<double> inputs)
	{
		for (int i = 0; i < inputNeurons.size() - 1; i++) 
		{
			inputNeurons[i].setValue(inputs[i]);
		}
		for (int i = 0; i < hideNeurons.size(); i++)
		{
			for (int j = 0; j < hideNeurons[i].size(); j++) 
			{
				double sum = 0.0;
				if (i == 0) 
				{
					for (int k = 0; k < inputNeurons.size(); k++)
					{
						sum += inputNeurons[k].getValue() * inputNeurons[k].getWeights()[j];
					}
				}
				else 
				{
					for (int k = 0; k < hideNeurons[i - 1].size(); k++)
					{
						sum += hideNeurons[i - 1][k].getValue() * hideNeurons[i - 1][k].getWeights()[j];
					}
				}
				if (hideNeurons[i][j].getBias())
				{
					hideNeurons[i][j].setValue(1);
				}
				else 
				{
					hideNeurons[i][j].setValue(activate(sum));
				}
				
			}
		}
		for (int i = 0; i < outputNeurons.size(); i++)
		{
			double sum = 0.0;
			for (int j = 0; j < hideNeurons[hideNeurons.size() - 1].size(); j++)
			{
				sum += hideNeurons[hideNeurons.size() - 1][j].getValue() * hideNeurons[hideNeurons.size() - 1][j].getWeights()[i];
			};
			outputNeurons[i].setValue(activate(sum));
		}
	}
	void backPropagation(std::vector<double> inputs, int value)
	{
		forwardPropagation(inputs);
		for (int i = 0; i < outputNeurons.size(); i++)
		{
			outputNeurons[i].setError(value - outputNeurons[i].getValue());
		}
		for (int i = 0; i < outputNeurons.size(); i++)
		{
			outputNeurons[i].setGradient(outputNeurons[i].getError() * derivative(outputNeurons[i].getValue()));
		}
		for (int i = hideNeurons.size() - 1; i > -1; i--)
		{
			for (int j = 0; j < hideNeurons[i].size(); j++)
			{
				double sum = 0.0;
				if (i == hideNeurons.size() - 1)
				{
					for (int k = 0; k < outputNeurons.size(); k++)
					{
						sum += outputNeurons[k].getError() * hideNeurons[i][j].getWeights()[k];
					}
				}
				else
				{
					for (int k = 0; k < hideNeurons[i + 1].size(); k++)
					{
						sum += hideNeurons[i + 1][k].getError() * hideNeurons[i][j].getWeights()[k];
					}
				}
				hideNeurons[i][j].setError(sum);
			}
		}

		for (int i = hideNeurons.size() - 1; i > -1; i--)
		{
			for (int j = 0; j < hideNeurons[i].size(); j++)
			{
				hideNeurons[i][j].setGradient(hideNeurons[i][j].getError() * derivative(hideNeurons[i][j].getValue()));
			}
		}
		for (int i = hideNeurons.size() - 1; i > -1 ; i--)
		{
			for (int j = 0; j < hideNeurons[i].size(); j++)
			{
				if (i == hideNeurons.size() - 1)
				{
					for (int k = 0; k < outputNeurons.size(); k++)
					{
						double deltaW = learningRate * outputNeurons[k].getGradient() * hideNeurons[i][j].getValue() + momentNes * hideNeurons[i][j].getsetLastDeltaWeights()[k];
						hideNeurons[i][j].setWeights(k, hideNeurons[i][j].getWeights()[k] + deltaW);
						hideNeurons[i][j].setLastDeltaWeights(k, deltaW);
					}
				}
				else
				{
					for (int k = 0; k < hideNeurons[i + 1].size(); k++)
					{
						double deltaW = learningRate * hideNeurons[i + 1][k].getGradient() * hideNeurons[i][j].getValue() + momentNes * hideNeurons[i][j].getsetLastDeltaWeights()[k];
						hideNeurons[i][j].setWeights(k, hideNeurons[i][j].getWeights()[k] + deltaW);
						hideNeurons[i][j].setLastDeltaWeights(k, deltaW);
					}
				}
			}
		}
		for (int i = 0; i < inputNeurons.size(); i++)
		{
			for (int j = 0; j < hideNeurons[0].size(); j++)
			{
				double deltaW = learningRate * hideNeurons[0][j].getGradient() * inputNeurons[i].getValue() + inputNeurons[i].getsetLastDeltaWeights()[j] * momentNes;
				inputNeurons[i].setWeights(j, inputNeurons[i].getWeights()[j] + deltaW);
				inputNeurons[i].setLastDeltaWeights(j, deltaW);
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
		matplot::plot(coordinatesX, coordinatesY);
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
			coordinatesY.push_back((1.0f / outputNeurons.size()) * sum);
			error = (1.0f / outputNeurons.size()) * sum;
			if (i > maxIteration)
			{
				break;
			}
			i++;
		}
		matplot::plot(coordinatesX, coordinatesY);
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
	std::vector<double> outputSet = { 0, 0, 1, 1 };
	NeuralNetwork nn(2, { 2 }, 1, 1, 0.8, true);
	nn.printResultTrain(inputSet);
	clock_t start = clock();
	nn.trainBeforeTheError(inputSet, outputSet, 0.01, 10000);
	//nn.trainToIterarion(inputSet, outputSet, 500);
	clock_t end = clock();
	std::cout << (end - start) / (CLOCKS_PER_SEC / 1000.0f) << "\n";
	nn.printResultTrain(inputSet);
	matplot::show();
	return 0;
}