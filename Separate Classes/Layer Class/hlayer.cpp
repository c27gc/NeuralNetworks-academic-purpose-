#include "hlayer.h"
#include "neuron.h"
#include <new>
#include <cstddef>
#include <iostream>
#include <math.h>  

HLayerClass::HLayerClass()
{
	this->numberofneurons=1;
	neuron =new(nothrow) NeuronClass[1];
	if (neuron == NULL)
	{
		cout<<"Error creating neurons, insufficient memory...";
	};
	this->inputslnumbers=1;
	this->inputlvectorPtr = new(nothrow) float[this->inputslnumbers];
	if (inputlvectorPtr == NULL)
	{
		cout << "Error creating input vector, insufficient memory...";
	};
	for (int i = 0; i < this->inputslnumbers; ++i)
	{
		inputlvectorPtr[i]=0;
	};
	neuron->setInputsValues(this->inputlvectorPtr);
	this->biaslweightsPtr = new(nothrow) float[this->numberofneurons];
	if (this->biaslweightsPtr==NULL)
	{
		cout << "Error creating bias weights vector, insufficient memory";
	};
	for (int i = 0; i < this->numberofneurons; ++i)
	{
		this->biaslweightsPtr[i] = 0;
	};
	for (int i = 0; i < this->numberofneurons; ++i)
	{	
		neuron[i].setBiasWeight(this->biaslweightsPtr[i]);
	};
	arrayofweights = new(nothrow) float*[this->inputslnumbers];
	if (this->arrayofweights == NULL)
	{
		cout << "Error creating array of weights [1], insufficient memory";
	};
	for (int i = 0; i < this->inputslnumbers; ++i)
	{
		arrayofweights[i] = new(nothrow) float[this->numberofneurons];
		if (this->arrayofweights[i] == NULL)
		{
			cout << "Error creating array of weights [2], insufficient memory";
		};
	};
	for (int i = 0; i < this->numberofneurons; ++i)
	{
		for (int j = 0; j < this->inputslnumbers; ++j)
		{
			arrayofweights[j][i]=0;
		};
	};
	float aux[this->inputslnumbers];
	for (int i = 0; i < this->numberofneurons; ++i)
	{
		for (int j = 0; j < this->inputslnumbers; ++j)
		{
			aux[j]=arrayofweights[j][i];
		};
		neuron[i].setInputsWeights(aux);
	};
}

HLayerClass::HLayerClass(int non, int iln, float *blw)
{
	this->numberofneurons=non;
	neuron =new(nothrow) NeuronClass[this->numberofneurons];
	if (neuron == NULL)
	{
		cout<<"Error creating neurons, insufficient memory...";
	};

	this->inputslnumbers=iln;
	this->inputlvectorPtr = new(nothrow) float[this->inputslnumbers];
	if (inputlvectorPtr == NULL)
	{
		cout << "Error creating input vector, insufficient memory...";
	};
	for (int i = 0; i < this->inputslnumbers; ++i)
	{
		inputlvectorPtr[i]=0;
	};
	neuron->setInputsValues(this->inputlvectorPtr);

	this->biaslweightsPtr = new(nothrow) float[this->numberofneurons];
	if (this->biaslweightsPtr==NULL)
	{
		cout << "Error creating bias weights vector, insufficient memory";
	};
	for (int i = 0; i < this->numberofneurons; ++i)
	{
		this->biaslweightsPtr[i] = blw[i];	
	};
	for (int i = 0; i < this->numberofneurons; ++i)
	{
		neuron[i].setBiasWeight(this->biaslweightsPtr[i]);
	};

	arrayofweights = new(nothrow) float*[this->inputslnumbers];
	if (this->arrayofweights == NULL)
	{
		cout << "Error creating array of weights [1], insufficient memory";
	};
	for (int i = 0; i < this->inputslnumbers; ++i)
	{
		arrayofweights[i] = new(nothrow) float[this->numberofneurons];
		if (this->arrayofweights[i] == NULL)
		{
			cout << "Error creating array of weights [2], insufficient memory";
		};
	};

	for (int i = 0; i < this->numberofneurons; ++i)
	{
		for (int j = 0; j < this->inputslnumbers; ++j)
		{
			arrayofweights[j][i]=0;
		};
	};
	float aux[this->inputslnumbers];
	for (int i = 0; i < this->numberofneurons; ++i)
	{
		for (int j = 0; j < this->inputslnumbers; ++j)
		{
			aux[j]=arrayofweights[j][i];
		};
		neuron[i].setInputsWeights(aux);
	};


	for (int i = 0; i < this->numberofneurons; ++i)
	{
		this->outputlvectorPtr[i]=0;
	};

}

HLayerClass::~HLayerClass()
{
	delete[] neuron;
	delete[] this->inputlvectorPtr;
	delete[] this->biaslweightsPtr;
	
	for (int i = 0; i < this->numberofneurons; ++i)
	{
		delete[] this->arrayofweights[i];
	}
	delete[] this->arrayofweights;
	delete[] this->outputlvectorPtr;
}

void HLayerClass::setNumberOfNeurons(int a)
{	
	this->numberofneurons=a;
}

void HLayerClass::createNeurons()
{
	delete[] neuron;
	neuron =new(nothrow) NeuronClass[this->numberofneurons];
	if (neuron == NULL)
	{
		cout<<"Error creating neurons, insufficient memory...";
	};
}

void HLayerClass::setLInputsNumbers(int b)
{
	this->inputslnumbers=b;
	for (int i = 0; i < this->numberofneurons; ++i)
	{
		neuron[i].setInputsNumbers(b);
	};
	
	
}

void HLayerClass::setLInputsValues(float *d)
{
	delete[] this->inputlvectorPtr;
	this->inputlvectorPtr = new(nothrow) float[this->inputslnumbers];
	if (inputlvectorPtr == NULL)
	{
		cout << "Error creating input vector, insufficient memory...";
	};
	for (int i = 0; i < this->inputslnumbers; ++i)
	{
		this->inputlvectorPtr[i]=d[i];
	};
	for (int i = 0; i < this->numberofneurons; ++i)
	{
		neuron[i].setInputsValues(this->inputlvectorPtr);
	};
	
}

void HLayerClass::setLBiasWeightsValues(float *k)
{
	delete[] this->biaslweightsPtr;
	this->biaslweightsPtr = new(nothrow) float[this->numberofneurons];
	if (this->biaslweightsPtr == NULL)
	{
		cout << "Error creating bias weights vector, insufficient memory";
	};
	for (int i = 0; i < this->numberofneurons; ++i)
	{
		this->biaslweightsPtr[i] = k[i];	
	};
	for (int i = 0; i < this->numberofneurons; ++i)
	{
		neuron[i].setBiasWeight(this->biaslweightsPtr[i]);
	};
}

void HLayerClass::getLBiasWeightsValues(float *k)
{
	for (int i = 0; i < this->numberofneurons; ++i)
	{
		this->biaslweightsPtr[i]=neuron[i].getBiasWeightValue();
		k[i]=this->biaslweightsPtr[i];
	};
}

void HLayerClass::setLWeightsValues(float **e)
{
	for (int i = 0; i < this->numberofneurons; ++i)
	{
		delete[] this->arrayofweights[i];
	}
	delete[] this->arrayofweights;

	arrayofweights = new(nothrow) float*[this->inputslnumbers];
	if (this->arrayofweights == NULL)
	{
		cout << "Error creating array of weights [1], insufficient memory";
	};
	for (int i = 0; i < this->inputslnumbers; ++i)
	{
		arrayofweights[i] = new(nothrow) float[this->numberofneurons];
		if (this->arrayofweights[i] == NULL)
		{
			cout << "Error creating array of weights [2], insufficient memory";
		};
	};

	for (int i = 0; i < this->numberofneurons; ++i)
	{
		for (int j = 0; j < this->inputslnumbers; ++j)
		{
			arrayofweights[j][i]=e[j][i];
		};
	};
	float aux[this->inputslnumbers];
	for (int i = 0; i < this->numberofneurons; ++i)
	{
		for (int j = 0; j < this->inputslnumbers; ++j)
		{
			aux[j]=arrayofweights[j][i];
		};
		neuron[i].setInputsWeights(aux);
	};
}

void HLayerClass::getLWeightsValues(float **e)
{	
	float aux[this->inputslnumbers];
	for (int i = 0; i < this->numberofneurons; ++i)
	{	
		neuron[i].getWeightsValues(aux);
		for (int j = 0; j < this->inputslnumbers; ++j)
		{
			this->arrayofweights[j][i]=aux[j];
			e[j][i]=this->arrayofweights[j][i];
		};
	};
}

float HLayerClass::getLIndividualWeightsValue(int nrn, int intp){return this->arrayofweights[intp][nrn];}

void HLayerClass::lInformationSpread()
{
	for (int i = 0; i < numberofneurons; ++i)
	{
		cout << "neuron: " << i << " " << endl;
		neuron[i].informationSpread();
	}
}

void HLayerClass::getLOutputsValues(float *c)
{
	for (int i = 0; i < numberofneurons; ++i)
	{
		c[i]=neuron[i].getOutputValue();
	}
}

float HLayerClass::getLOutputIndividualValue(int ndn){return this->neuron[ndn].getOutputValue();}