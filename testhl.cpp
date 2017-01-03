#include <iostream>
#include <string.h>
#include <math.h> 
#include "neuralnetwork.h"
using namespace std;

int main(int argc, char const *argv[])
{
	HLayerClass *Layer;
	Layer = new(nothrow) HLayerClass[1];
	if (Layer == NULL)
	{
		cout << "Error creando Layer" << endl;
	};
	
	Layer[0].setNumberOfNeurons(3);
	Layer[0].createNeurons();
	Layer[0].setLInputsNumbers(2);

	float *inputv;
	inputv = new float[2];
	inputv[0]=1;
	inputv[1]=1;

	Layer[0].setLInputsValues(inputv);

	float biasv[3];
	biasv[0]=0.5;
	biasv[1]=0.5;
	biasv[2]=0.5;

	Layer[0].setLBiasWeightsValues(biasv);

	float *biasv2;
	biasv2 = new float[3];

	Layer[0].getLBiasWeightsValues(biasv2);

	for (int i = 0; i < 3; ++i)
	{
		cout << "Bias[" << i << "]: " << biasv2[i] << endl;
	};

	float **weights;

	weights = new(nothrow) float*[2];
	if (weights == NULL)
	{
		cout << "Error creating array of weights [1], insufficient memory";
	};
	for (int i = 0; i < 2; ++i)
	{
		weights[i] = new(nothrow) float[3];
		if (weights[i] == NULL)
		{
			cout << "Error creating array of weights [2], insufficient memory";
		};
	};

	weights[0][0]=0;
	cout << "weights 1: " << endl;
	for (int i = 0; i < 2; ++i)
	{
		for (int j = 0; j < 3; ++j)
		{	
			weights[i][j]=0.5;
			cout << weights[i][j] << "   " ;
		};
		cout << endl;
	};
	cout << "aqui" << endl;
	Layer[0].setLWeightsValues(weights);

	float **weights2;

	weights2 = new(nothrow) float*[2];
	if (weights2 == NULL)
	{
		cout << "Error creating array of weights2 [1], insufficient memory";
	};
	for (int i = 0; i < 2; ++i)
	{
		weights2[i] = new(nothrow) float[3];
		if (weights2[i] == NULL)
		{
			cout << "Error creating array of weights [2], insufficient memory";
		};
	};

	Layer[0].getLWeightsValues(weights2);

	cout << "weights 2: " << endl;
	for (int i = 0; i < 2; ++i)
	{
		for (int j = 0; j < 3; ++j)
		{
			cout << weights2[i][j] << "   ";
		}
		cout << endl;
	};

	Layer[0].getLIndividualWeightsValue(2,1);

	float *outputs;
	outputs = new float[3];

	Layer[0].getLOutputsValues(outputs);

	for (int i = 0; i < 3; ++i)
	{
		cout << "output[" << i << "] :" << outputs[i] << endl;
	};

	Layer[0].lInformationSpread();
	Layer[0].getLOutputsValues(outputs);
	cout << endl;
	for (int i = 0; i < 3; ++i)
	{
		cout << "output[" << i << "] :" << outputs[i] << endl;
	};
}