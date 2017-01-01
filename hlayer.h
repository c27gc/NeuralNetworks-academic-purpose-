#ifndef hlayer_H
#define hlayer_H

#include "neuron.h"
using namespace std;

class HLayerClass
{
	public:
		HLayerClass();
		HLayerClass(int a, int b, float *d, float *e);
		~HLayerClass();
		void setNumberOfNeurons(int a);
		void createNeurons();
		void setLInputsNumbers(int b);
		void setLInputsValues(float *d);
		void setLBiasWeightsValues(float *k);
		void getLBiasWeightsValues(float *k);
		void setLWeightsValues(float *e);
		void getLWeightsValues(float *e);
		void lInformationSpread();
		void getLOutputsValues(float *c);
		int getLOutputsNumber();
	private:
		int numberofneurons;
		NeuronClass *neuron;
		int inputslnumbers;
		float *inputlvectorPtr;
		float *biaslweightsPtr;
		float **arrayofweights;
		float *outputlvectorPtr;
};
#endif