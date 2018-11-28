#include "Layer.h"



Layer::Layer(unsigned int Row, unsigned int Col)
	:
	Number_Of_Neurons_(Row * Col),
	Row_(Row),
	Col_(Col),
	Sheet_(1)
{
	this->Layer_Classification_ = new std::string("Input layer");
	this->Output_				= new GMat::shared_vector(Row, Col, 1, Zeroes);
	this->Stack_Of_Imgs_		= nullptr;
}

Layer::Layer(unsigned int Row, unsigned int Col, unsigned int Sheet, const std::string& layer_classification, float learning_rate, float bias, ElementValues elem)
	:
	Number_Of_Neurons_(Row * Col * Sheet),
	Sheet_(Sheet),
	Row_(Row),
	Learning_Rate_(learning_rate),
	Bias_(bias)
{
	this->Layer_Classification_ = new std::string(layer_classification);
	this->Output_				= new GMat::shared_vector(Row, Col, Sheet, elem);
	this->Stack_Of_Imgs_		= nullptr;
}

Layer::~Layer()
{

}

unsigned int Layer::GetNumberOfSheets() const
{
	return this->Sheet_;
}

unsigned int Layer::GetNumberOfNeurons() const
{
	return this->Number_Of_Neurons_;
}

void Layer::ResizeOutput(unsigned int Sheet)
{
	this->Stack_Of_Imgs_ = new GMat::shared_vector(1);
	for (unsigned int i = 0; i < Sheet; i++)
	{
		this->Stack_Of_Imgs_->push_back(GMat::Matrix(this->Row_, this->Col_, Zeroes));
	}
	this->Sheet_ = Sheet;
}

GMat::shared_vector& Layer::GetOutputMatrix()
{
	return *this->Output_;
}

GMat::shared_vector& Layer::GetStackOfImgs()
{
	return *this->Stack_Of_Imgs_;
}

std::string& Layer::LayerClassification() const
{
	return *this->Layer_Classification_;
}
