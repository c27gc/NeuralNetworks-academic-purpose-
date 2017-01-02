#include <iostream>
#include <string.h>
#include <math.h> 
#include "neuron.h"
using namespace std;


int main() {

	NeuronClass neuron;
	cout << "--------------Prueba de Constructor Predeterminado--------------" << endl;
	cout << "input numbers: " << neuron.getInputsNumber() << endl;
	cout << "weights : ";
	float a[neuron.getInputsNumber()];
	neuron.getWeightsValues(a);
	for (int i = 0; i < neuron.getInputsNumber(); ++i)
	{
		cout << "      Peso " << i << ": " << a[i] << endl;
	};
	cout << "BiasWeight: " << neuron.getBiasWeightValue() << endl;
	cout<< "Output: " << neuron.getOutputValue() << endl;
	cout << "-----------------Prueba de Constructor Definido-----------------" << endl;
	float b[5], c[5];
	for (int i = 0; i < 5; ++i)
	{
		b[i]=i/5.0;
		c[i]=i/10.0;

	};

	NeuronClass neuron2(5,b,0.5);
	neuron2.setInputsValues(c);
	cout<< "Valores antes de la Propagación: " << endl;

	cout << "	input numbers: " << neuron2.getInputsNumber() << endl;
	cout << "	weights : " << endl;
	float a1[neuron2.getInputsNumber()];
	neuron2.getWeightsValues(a1);

	for (int i = 0; i < neuron2.getInputsNumber(); ++i)
	{
		cout << "		Peso " << i << ": " << a1[i] << endl;
	};

	cout << "	BiasWeight: " << neuron2.getBiasWeightValue() << endl;
	cout<< "	Output: " << neuron2.getOutputValue() << endl;
	neuron2.informationSpread();
	cout<< "Valores despues de la Propagación: " << endl;

	cout << "	input numbers: " << neuron2.getInputsNumber() << endl;
	cout << "	weights : " << endl;
	float a2[neuron2.getInputsNumber()];
	neuron2.getWeightsValues(a2);
	for (int i = 0; i < neuron2.getInputsNumber(); ++i)
	{
		cout << "		Peso " << i << ": " << a2[i] << endl;
	};
	cout << "	BiasWeight: " << neuron2.getBiasWeightValue() << endl;
	cout<< "	Output: " << neuron2.getOutputValue() << endl;
	neuron2.informationSpread();
	cout << "------------------Prueba de Set Inputs Weights------------------" << endl;
	NeuronClass neuron3;
	neuron3.setInputsNumbers(3);
	cout << "weights : " << endl;
	float a4[neuron.getInputsNumber()];
	neuron3.getWeightsValues(a4);
	for (int i = 0; i < neuron3.getInputsNumber(); ++i)
	{
		cout << "      Peso " << i << ": " << a4[i] << endl;
	};

	neuron3.setInputsWeights(b);
	cout << "Despues de settear inputs: " << endl;
	cout << "weights : " << endl;
	float a5[neuron.getInputsNumber()];
	neuron3.getWeightsValues(a5);
	for (int i = 0; i < neuron3.getInputsNumber(); ++i)
	{
		cout << "      Peso " << i << ": " << a5[i] << endl;
	};

	neuron3.informationSpread();
	cout << "Output: " << neuron3.getOutputValue() << endl;


	//cout<< "Valor de salida despues de propagación: " << neuron.getOutputValue() << endl;
};
