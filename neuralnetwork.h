#ifndef neuron_H
#define neuron_H
//--------------------Class Definition--------------------//

class NeuronClass
{
	public:
		NeuronClass();
		NeuronClass(int x, float *b, float c);
		~NeuronClass();
		void setInputsNumbers(int b);
		void setInputsWeights(float *b);
		void setBiasWeight(float c);
		void setInputsValues(float *b);
		void informationSpread();
		float getOutputValue();
		int getInputsNumber();
		float getBiasWeightValue();
		void getWeightsValues(float *k);
	private:
		int inputsnumber;
		float *inputsweightsPtr;
		float *inputsvaluesPtr;
		float *biasweight;
		float outputvalue;
};

#endif

#ifndef hlayer_H
#define hlayer_H


class HLayerClass
{
	public:
		HLayerClass();
		HLayerClass(int non, int iln, float *blw);
		~HLayerClass();
		void setNumberOfNeurons(int a);
		int getNumberOfNeurons();
		void createNeurons();
		void setLInputsNumbers(int b);
		void getLInputsNumbers();
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
		int auxinputslnumbers;
		float *inputlvectorPtr;
		float *biaslweightsPtr;
		float **arrayofweights;
		float *outputlvectorPtr;
};
#endif

#ifndef network_H
#define network_H

class NetworkClass
{
	public:
		NetworkClass();
		NetworkClass(int nol, int *nonpl);
		~NetworkClass();
		void setNumberOfLayers(int t);
		void setNumberOfNeuronsPerLayer(int *f);
		void setInputsNValues(int *g);
		void setNWeightsValues(int ***we);
		//void backpropagationlearning();
		void nInformationSpread();
	private:
		bool aux1 = true;
		bool aux2 = false;
		int numberoflayers;
		int numberoflayersaux;
		HLayerClass *layer;
		int *numberofneuronsperlayerPtr;
		int numberofoutputs;
		float *desiredoutputsvaluesPtr;
		int numberofinputs;
		float *inputsnvaluesPtr;
		float *outputsnvaluesPtr;
		float ***arrayNofweights;
};

#endif
