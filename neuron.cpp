#include "neuron.h"
#include <new>
#include <cstddef>
#include <iostream>
#include <math.h>  

using namespace std;

NeuronClass::NeuronClass()
{

	this->inputsnumber = 10;

	this->inputsweightsPtr = new (nothrow) float[this->inputsnumber];
	if (inputsweightsPtr == NULL)
	{
		cout<<"Error initializing weights in 0...";
	};
	for (int i = 0; i < this->inputsnumber; ++i)
	{
		this->inputsweightsPtr[i]=0;
	}

	this->inputsvaluesPtr = new (nothrow) float[this->inputsnumber];
	if (inputsvaluesPtr == NULL)
	{
		cout<<"Error initializing inputs in 0...";
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
		cout<<"Error initializing weights in 0...";
	};
	for (int i = 0; i < this->inputsnumber; ++i)
	{
		this->inputsweightsPtr[i]=b[i];
	};

	this->inputsvaluesPtr = new (nothrow) float[this->inputsnumber];
	if (inputsvaluesPtr == NULL)
	{
		cout<<"Error initializing inputs in 0...";
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
		cout<<"Error setting weights...";
	};
	for (int i = 0; i < inputsnumber; ++i)
	{
		this->inputsweightsPtr[i]=b[i];
	};

}

void NeuronClass::setBiasWeight(float c)
{	
	cout << "aqui llega: " << c << endl; 
	this->biasweight=c;
}


void NeuronClass::setInputsValues(float *b)
{
	delete[] this->inputsvaluesPtr;
	this->inputsvaluesPtr = new (nothrow) float[this->inputsnumber];
	if (inputsvaluesPtr == NULL)
	{
		cout<<"Error setting inputs values...";
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