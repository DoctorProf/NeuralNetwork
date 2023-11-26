#pragma once
#include "Neuron.h"
#include "matplot/matplot.h"
#include <cmath>
#include <stdlib.h>

class NeuralNetwork
{
private:
	bool useBias;
	double momentNes;
	double learningRate;
	std::vector<std::vector<Neuron>> layers;
	std::vector<int> coordinatesX;
	std::vector<double> coordinatesY;

	double activate(double x);
	double derivative(double x);
public:
	NeuralNetwork(int countInputNeuron, std::vector<int> countHideNeuron, int countOutputNeuron, double learningRate, double momentNes, bool useBias);
	double getResult();
	double getErrorSquare();
	double getError();
	void forwardPropagation(std::vector<double> inputs);
	void backPropagation(std::vector<double> inputs, double value);
	void trainToIterarion(std::vector<std::vector<double>> inputSet, std::vector<double> outputSet, int iteration);
	void trainBeforeTheError(std::vector<std::vector<double>> inputSet, std::vector<double> outputSet, double errorMax, int maxIteration);
	void printResultTrain(std::vector<std::vector<double>> inputSet);
	void saveWeights();
};