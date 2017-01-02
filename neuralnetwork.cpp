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
		this->neuron[i].setInputsWeights(aux);
	};


	for (int i = 0; i < this->numberofneurons; ++i)
	{
		this->outputlvectorPtr[i]=0;
	};

}

HLayerClass::HLayerClass(int non, int iln, float *blw)
{
	this->numberofneurons=non;
	NeuronClass *neuron;
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
		this->neuron[i].setInputsWeights(aux);
	};


	for (int i = 0; i < this->numberofneurons; ++i)
	{
		this->outputlvectorPtr[i]=0;
	};

}

HLayerClass::~HLayerClass()
{
	delete[] this->neuron;
	delete[] this->inputlvectorPtr;
	delete[] this->biaslweightsPtr;
	delete[] this->arrayofweights;
	delete[] this->outputlvectorPtr;
}

void HLayerClass::setNumberOfNeurons(int a){this->numberofneurons=a;}

void HLayerClass::createNeurons()
{
	NeuronClass *neuron;
	neuron =new(nothrow) NeuronClass[this->numberofneurons];
	if (neuron == NULL)
	{
		cout<<"Error creating neurons, insufficient memory...";
	};
}

void HLayerClass::setLInputsNumbers(int b){this->inputslnumbers=b;}

void HLayerClass::setLInputsValues(float *d)
{
	this->inputlvectorPtr = new(nothrow) float[this->inputslnumbers];
	if (inputlvectorPtr == NULL)
	{
		cout << "Error creating input vector, insufficient memory...";
	};
	for (int i = 0; i < this->inputslnumbers; ++i)
	{
		inputlvectorPtr[i]=d[i];
	};
	neuron->setInputsValues(this->inputlvectorPtr);
}

void HLayerClass::setLBiasWeightsValues(float *k)
{
	this->biaslweightsPtr = new(nothrow) float[this->numberofneurons];
	if (this->biaslweightsPtr==NULL)
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
		this->neuron[i].setInputsWeights(aux);
	};
}

void HLayerClass::getLWeightsValues(float **e)
{	
	float aux[this->inputslnumbers];
	for (int i = 0; i < this->numberofneurons; ++i)
	{	
		neuron[i].getWeightsValues(float aux);
		for (int j = 0; j < this->inputslnumbers; ++j)
		{
			this->arrayofweights[j][i]=aux[j];
			b[j][i]=this->arrayofweights[j][i];
		};
	};
}

float getLIndividualWeightsValue(int nrn, int intp){return this->arrayofweights[intp-1][nrn-1]}