#include "Convnet.h"
#include <PerformanceCheck.h>


int main()
{

	int total_training, sub_training, print_num, avg, imgs_per_bp;
	std::cin >> total_training >> sub_training >> imgs_per_bp >> print_num >> avg;

	Convnet* cnn = new Convnet();
	cnn->SetImage(Mnist_);
	cnn->Training(total_training, sub_training, imgs_per_bp, print_num, avg);
	cnn->Simulation(1);


	return 0;
}