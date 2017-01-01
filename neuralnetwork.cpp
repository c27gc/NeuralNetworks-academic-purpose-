#include "neuralnetwork.h"
#include <new>
#include <cstddef>
#include <iostream>
#include <math.h>  

using namespace std;

NeuronClass::NeuronClass()
{

	this->inputsnumber = 1;

	this->inputsweightsPtr = new (nothrow) float[this->inputsnumber];
	if (inputsweightsPtr == NULL)
	{
		cout<<"Error creating weights, insufficient memory...";
	};
	for (int i = 0; i < this->inputsnumber; ++i)
	{
		this->inputsweightsPtr[i]=0;
	}

	this->inputsvaluesPtr = new (nothrow) float[this->inputsnumber];
	if (inputsvaluesPtr == NULL)
	{
		cout<<"Error creating inputs, insufficient memory...";
	};
	for (int i = 0; i < this->inputsnumber; ++i)
	{
		this->inputsvaluesPtr[i]=0;
	};

	this->biasweight=0;

	this->outputvalue=0;
}

NeuronClass::NeuronClass(int x, float *b, float c)
{	

	this->inputsnumber=x;

	this->inputsweightsPtr = new (nothrow) float[x];
	if (inputsweightsPtr == NULL)
	{
		cout<<"Error creating weights, insufficient memory...";
	};
	for (int i = 0; i < this->inputsnumber; ++i)
	{
		this->inputsweightsPtr[i]=b[i];
	};

	this->inputsvaluesPtr = new (nothrow) float[this->inputsnumber];
	if (inputsvaluesPtr == NULL)
	{
		cout<<"Error creating inputs, insufficient memory...";
	};
	for (int i = 0; i < this->inputsnumber; ++i)
	{
		this->inputsvaluesPtr[i]=0;
	};
	
	this->biasweight=c;
	
	this->outputvalue=0;
}

NeuronClass::~NeuronClass(){
	delete[] this->inputsweightsPtr;
	delete[] this->inputsvaluesPtr;

}

void NeuronClass::setInputsNumbers(int b)
{
	this->inputsnumber=b;
}

void NeuronClass::setInputsWeights(float *b)
{
	delete[] this->inputsweightsPtr;
	this->inputsweightsPtr = new (nothrow) float[this->inputsnumber];
	if (inputsweightsPtr == NULL)
	{
		cout<<"Error setting weights, insufficient memory...";
	};
	for (int i = 0; i < inputsnumber; ++i)
	{
		this->inputsweightsPtr[i]=b[i];
	};

}

void NeuronClass::setBiasWeight(float c)
{
	this->biasweight=c;
}


void NeuronClass::setInputsValues(float *b)
{
	delete[] this->inputsvaluesPtr;
	this->inputsvaluesPtr = new (nothrow) float[this->inputsnumber];
	if (inputsvaluesPtr == NULL)
	{
		cout<<"Error setting inputs values, insufficient memory...";
	};
	for (int i = 0; i < this->inputsnumber; ++i)
	{
		this->inputsvaluesPtr[i]=b[i];
	};

}

void NeuronClass::informationSpread()
{
	float u;

	u=this->biasweight;

	for (int i = 0; i < this->inputsnumber; ++i)
	{
		u+=(this->inputsvaluesPtr[i]) * (this->inputsweightsPtr[i]);

	}

	this->outputvalue=(1)/(1+(exp(-u)));
}

float NeuronClass::getOutputValue(){return this->outputvalue;}

int NeuronClass::getInputsNumber(){return this->inputsnumber;}

float NeuronClass::getBiasWeightValue(){return this->biasweight;}

void NeuronClass::getWeightsValues(float *k)
{
	for (int i = 0; i < this->inputsnumber; ++i)
	{
		k[i]=this->inputsweightsPtr[i];
	};
}

HLayerClass::HLayerClass()
{
	this->numberofneurons=1;
	NeuronClass *neuron;
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
		this->biaslweightsPtr = 0;	
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

	for (int i = 0; i < this->numberofneurons; ++i)
	{
		this->outputlvectorPtr[i]=0;
	};


}