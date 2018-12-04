#include "Convnet.h"
#include <PerformanceCheck.h>


int main()
{

	/*int total_training, sub_training, print_num, avg, imgs_per_bp;
	std::cin >> total_training >> sub_training >> imgs_per_bp >> print_num >> avg;*/

	Convnet* cnn = new Convnet();
	/*cnn->SetImage(Mnist_);
	cnn->Training(total_training, sub_training, imgs_per_bp, print_num, avg);
	cnn->Simulation(1);*/

	float LR = LearningRate;
	FSM
	{
		STATE(step_one) //load 20,000 images for training.
			cnn->SetImage(Mnist_);
		NEXT_STATE(step_two)

		STATE(step_two) //train on 20,000 images.
			cnn->Training(1,2000,10,200,10);
		NEXT_STATE(step_three)

		STATE(step_three)
			cnn->Simulation(ImageTestLimit);
		NEXT_STATE(step_four)

		STATE(step_four)
			if (cnn->TotalError() == true) //which mean the training is completed.
			{
				NEXT_STATE(step_six)
			}
			else
			{
				NEXT_STATE(step_five) // reduce the learning rate.
			}

		STATE(step_five)
			LR /= 5;
			if (LR > 0.000001f)
			{
				cnn->SetLearningRate(LR);
			}
			NEXT_STATE(step_one)

		STATE(step_six)
			cnn->SaveWeightsToDataFile();
	}
	return 0;
}