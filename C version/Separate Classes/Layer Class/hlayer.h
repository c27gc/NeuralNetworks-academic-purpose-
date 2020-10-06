#ifndef hlayer_H
#define hlayer_H

using namespace std;

class HLayerClass
{
	public:
		HLayerClass();
		HLayerClass(int non, int iln, float *blw);
		~HLayerClass();
		void setNumberOfNeurons(int a);
		void createNeurons();
		void setLInputsNumbers(int b);
		void setLInputsValues(float *d);
		void setLBiasWeightsValues(float *k);
		void getLBiasWeightsValues(float *k);
		void setLWeightsValues(float **e);
		void getLWeightsValues(float **e);
		float getLIndividualWeightsValue(int nrn, int intp);
		void lInformationSpread();
		void getLOutputsValues(float *c);
		float getLOutputIndividualValue(int ndn);

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