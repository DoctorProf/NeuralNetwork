#pragma once
#include "Neuron.h"
#include "matplot/matplot.h"
#include <cmath>
#include <stdlib.h>
#include <fstream>
#include <algorithm>

class NeuralNetwork
{
private:
	bool useBias;
	double momentNes;
	double learningRate;
	int inputNeurons;
	std::vector<int> hideNeurons;
	int outputNeurons;
	std::vector<std::vector<Neuron>> layers;
	std::vector<int> coordinatesX;
	std::vector<double> coordinatesY;

	double activate(double x);
	double derivative(double x);
	void generateNeurons();
	void forwardPropagation(std::vector<double> inputs);
	void backPropagation(std::vector<double> inputs, std::vector<double> value);
	void initializeWeights();
	double getErrorSquare(int index);
	void saveNN();
public:
	NeuralNetwork(int inputNeurons, std::vector<int> hideNeurons, int outputNeurons, double learningRate, double momentNes, bool useBias);
	NeuralNetwork(std::string file);
	std::vector<std::vector<Neuron>> getLayers();
	void trainToIterarion(std::vector<std::vector<double>> inputSet, std::vector<std::vector<double>> outputSet, int iteration, bool save);
	void trainBeforeTheError(std::vector<std::vector<double>> inputSet, std::vector<std::vector<double>> outputSet, double errorMax, int maxIteration, bool save);
	void printResultTrain(std::vector<std::vector<double>> inputSet, std::vector<std::vector<double>> outputSet);
	void predict(std::vector<double> input);
};