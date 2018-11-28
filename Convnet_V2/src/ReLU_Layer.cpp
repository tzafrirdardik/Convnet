#include "Layer.h"


ReLU_Layer::ReLU_Layer(unsigned int Row, unsigned int Col, Layer* Prev_layer, unsigned int Sheet, float learning_rate, float bias)
	:
	Layer(Row, Col, Sheet, "ReLU layer", learning_rate, bias, Random_Val)
{
	this->Prev_layer_ = Prev_layer;
}

ReLU_Layer::~ReLU_Layer()
{

}

void ReLU_Layer::Simulate()
{
	for (unsigned int i = 0; i < this->GetNumberOfSheets(); i++)
		this->Output_->at(i)->ReLUMatrix(*this->Prev_layer_->GetOutputMatrix().at(i), 0);
}
