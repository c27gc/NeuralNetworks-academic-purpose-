#ifndef hlayer_H
#define hlayer_H

#include "neuron.h"
using namespace std;

class HLayerClass
{
	public:
		HLayerClass();
		HLayerClass(int a, int b);
		~HLayerClass();
		void createNeurons();
		void informationSpread();
		void getOutputs(float *c);
		void setWeights(float *d);
		void getWeights(float *e);
	private:
		NeuronClass *neuron;
		numberofneurons;
		numberofinputs;
		float *inputvectorPtr;
		float arrayofweights[numberofneurons][numberofinputs];
		float *outputvectorPtr;
}
#endif