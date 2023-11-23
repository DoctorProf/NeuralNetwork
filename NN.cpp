// 0.21v No Stable
#include <iostream>
#include <vector>
#include <cmath>

class Neuron 
{
private:
	std::vector<double> weights;
	bool toBias;
	double value;
	double error;
	double gradient;

	void initializeWeights() 
	{
		for (size_t i = 0; i < weights.size(); i++) 
		{
			weights[i] = (rand() % 100) / 50.0 - 1;
		}
	}
public:
	Neuron(int countWeight = 0, double value = 0, bool toBias = false) 
	{
		this->value = value;
		this->weights.resize(countWeight);
		this->toBias = toBias;
		initializeWeights();
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
				inputNeurons.push_back(Neuron(countHideNeuron[0] + 1));
			}
		}
		if (useBias) inputNeurons.push_back(Neuron(countHideNeuron[0] + 1, 1, true));
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
					hideNeurons[i].push_back(Neuron(countHideNeuron[i + 1] + 1));
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
					hideNeurons[i].push_back(Neuron(countHideNeuron[i + 1] + 1, 1, true));
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
		//std::cout << hideNeurons[0][0].getValue() << inputNeurons[0].getValue();
		return outputNeurons[0].getValue();
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
						hideNeurons[i][j].setWeights(k, hideNeurons[i][j].getWeights()[k] + learningRate * outputNeurons[k].getGradient() * hideNeurons[i][j].getValue());
					}
				}
				else
				{
					for (int k = 0; k < hideNeurons[i + 1].size(); k++)
					{
						hideNeurons[i][j].setWeights(k, hideNeurons[i][j].getWeights()[k] + learningRate * hideNeurons[i + 1][k].getGradient() * hideNeurons[i][j].getValue());
					}
				}
			}
		}
		for (int i = 0; i < inputNeurons.size(); i++)
		{
			for (int j = 0; j < hideNeurons[0].size(); j++)
			{
				inputNeurons[i].setWeights(j, inputNeurons[i].getWeights()[j] + learningRate * hideNeurons[0][j].getGradient() * inputNeurons[i].getValue());
			}
		}
	}
};
int main()
{
	std::vector<std::vector<double>> inputSet = { {0, 0}, {0, 1}, {1, 0}, {1, 1} };
	std::vector<double> outputSet = { 1, 0, 0, 0 };
	NeuralNetwork nn(2, { 2 }, 1, 5, 0.5, true);
	int iteration = 10000;
	for (int i = 0; i < iteration; i++)
	{
		for (int j = 0; j < inputSet.size(); j++) 
		{
			nn.backPropagation(inputSet[j], outputSet[j]);
			if (i == iteration - 1)
			{
				std::cout << j + 1 << " " << nn.getResult() << "\n";
			}
		}
	}
	return 0;
}