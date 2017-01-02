#ifndef neuron_H
#define neuron_H
//--------------------Class Definition--------------------//
using namespace std;

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
		float biasweight;
	private:
		int inputsnumber;
		float *inputsweightsPtr;
		float *inputsvaluesPtr;
		float outputvalue;
};

#endif
