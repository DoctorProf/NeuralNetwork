#include "../Headers/Neuron.h"

void Neuron::initializeWeights()
{
	srand(time(NULL));
	for (size_t i = 0; i < weights.size(); i++)
	{
		weights[i] = (rand() % 100) / 50.0 - 1;
	}
}
void Neuron::initializeDeltaWeights()
{
	for (size_t i = 0; i < lastDeltaWeights.size(); i++)
	{
		lastDeltaWeights[i] = 0;
	}
}
Neuron::Neuron(int countWeight, double value, bool toBias)
{
	this->value = value;
	this->weights.resize(countWeight);
	this->lastDeltaWeights.resize(countWeight);
	this->toBias = toBias;
	initializeWeights();
	initializeDeltaWeights();
}
void Neuron::setValue(double value)
{
	this->value = value;
}
double Neuron::getValue()
{
	return this->value;
}
void Neuron::setError(double error)
{
	this->error = error;
}
double Neuron::getError()
{
	return this->error;
}
void Neuron::setGradient(double gradient)
{
	this->gradient = gradient;
}
double Neuron::getGradient()
{
	return this->gradient;
}
void Neuron::setWeights(int index, double value)
{
	this->weights[index] = value;
}
std::vector<double> Neuron::getWeights()
{
	return this->weights;
}
void Neuron::setLastDeltaWeights(int index, double value)
{
	this->lastDeltaWeights[index] = value;
}
std::vector<double> Neuron::getLastDeltaWeights()
{
	return this->lastDeltaWeights;
}
bool Neuron::getBias()
{
	return this->toBias;
}