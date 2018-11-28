#ifndef _CONVOLUTIONAL_NEURAL_NET_WORK_
#define _CONVOLUTIONAL_NEURAL_NET_WORK_

#include "Layer.h"

class Convnet
{
public:

	Convnet();
	Convnet(float Learning_rate, float Bias);
	~Convnet();

	unsigned int GetNumberOfLayers() const;

	void Addlayer();
	void Eraselayer();

	void SetImage(WhichData_ data);
	void SetBias(float Val);
	void SetLearningRate(float Val);
	void SetActivationFunction(unsigned int Idx, unsigned int LayerIdx);
	void ShafelWeights();
	std::vector<float>& GetDataLabel();

	void FeedForward();
	void BackPropagation();
	void SetChangeInWeights(int average);
	void Training(unsigned int NumberOfEpoch, unsigned int NumberOfTrainingPerEpoch, unsigned int NumberOfImgPerBP, unsigned int PrintTime, int average = 1);
	void Simulation(unsigned int NumOfSimulations);
	void SaveWeightsToDataFile();
	void LoadWeightsFromDataFile();
	void GenerateExpectedOutput(const std::vector<float>& numbers);

	void PrintConvnetTopology() const;
	void PrintResult(const std::vector<float>& number, unsigned int epoch = 1, unsigned int Subepoch = 1) const;
	void PrintWeights() const;

protected:

	unsigned int				Size_;
	std::vector<Layer*>*		Layer_list_;
	std::vector<std::string>*	Topology_;
	float						Learning_rate_;
	GMat::Matrix*				Error_Vec_;
	GMat::shared_vector*		Expected_Output_Vec_;
	std::vector<float>*			Data_labels_;
};

#endif