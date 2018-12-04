#include "Layer.h"



Input_layer::Input_layer(unsigned int Row, unsigned int Col, WhichData_ WD)
	:
	Layer(Row, Col),
	WD_(WD)
{

}

Input_layer::~Input_layer()
{

}

void Input_layer::SetImage(const std::string& file_dir_)
{
	std::ifstream file;
	if (file.is_open())
	{
		//TODO: reading images from given folder.
	}
}

void Input_layer::Load_Mnist_Data(const std::string& file_dir, int Number_of_img)
{
	this->ResizeOutput(Number_of_img);
	GMat::read_Mnist(file_dir, this->GetStackOfImgs(), Number_of_img);
}

void Input_layer::Simulate()
{
	this->Output_->at(0)->operator=(*this->Stack_Of_Imgs_->back());
	this->Stack_Of_Imgs_->pop_back();
}

void Input_layer::SaveWeight()
{

}