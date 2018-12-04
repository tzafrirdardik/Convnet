#include "Layer.h"


Pool_layer::Pool_layer()
{
	this->Input_				= nullptr;
	this->Derivative_Output_	= nullptr;
	this->Weight_Matrix_		= nullptr;
	this->Delta_Weight_Matrix_	= nullptr;
	this->Prev_layer_			= nullptr;
}

Pool_layer::Pool_layer(unsigned int Row, unsigned int Col, Layer* Prev_layer, unsigned int Sheet, float learning_rate, float bias)
	:
	Layer(Row, Col, Sheet, "Pool layer", learning_rate, bias, Zeroes)
{
	this->Prev_layer_			= Prev_layer;
	this->Weight_Matrix_		= new GMat::Matrix(Sheet, 1, Random_Val);
	this->Delta_Weight_Matrix_	= new GMat::Matrix(Sheet, 1, Zeroes);
	this->Derivative_Output_	= new GMat::shared_vector(Row, Col, Sheet, Zeroes);
	this->Input_				= new GMat::shared_vector(Row, Col, Sheet, Zeroes);
}


Pool_layer::~Pool_layer()
{

}

GMat::Matrix* Pool_layer::GetWeightMatrix() const
{
	return this->Weight_Matrix_;
}

void Pool_layer::CalculateNewWeights(int average)
{
	(*this->Weight_Matrix_) += ((*this->Delta_Weight_Matrix_) / (float)average);
	this->Weight_Matrix_->Norm();
	this->Delta_Weight_Matrix_->SetMatrixToZero();
}

void Pool_layer::Simulate()
{
	for (unsigned int i = 0; i < this->GetNumberOfSheets(); i++)
	{
		this->Input_->at(i)->AVGPoolMatrix(*this->Prev_layer_->GetOutputMatrix().at(i));
		this->Output_->at(i)->ActivateMatrix((*this->Input_->at(i)) * (*this->Weight_Matrix_)(i, 0), 1, this->Bias_);
		this->Derivative_Output_->at(i)->DerivativeMatrix((*this->Input_->at(i)) * (*this->Weight_Matrix_)(i, 0), 1, this->Bias_);
	}
}

void Pool_layer::SaveWeight()
{
	unsigned int r_size = this->Weight_Matrix_->GetRowSize();
	unsigned int c_size = this->Weight_Matrix_->GetColSize();
	std::string tmp(STR(Pool_layer_weights_size_));
	tmp.append(STR(r_size));
	tmp.append(STR(x));
	tmp.append(STR(c_size));
	tmp.append(STR(.txt));
	this->Weight_Matrix_->SaveMatrix(tmp);
}

void Pool_layer::BP(Layer& layer_i, Layer& layer_k, GMat::unique_vector& gradient_k)
{
	if (layer_i.LayerClassification().compare("ReLU layer") == 0 && layer_k.LayerClassification().compare("FC layer") == 0)
	{

	}
	else if (layer_i.LayerClassification().compare("Conv layer") == 0 && layer_k.LayerClassification().compare("FC layer") == 0)
	{
		// I < J (current layer) < K
		FC_layer* K = dynamic_cast<FC_layer*>(&layer_k);
		Conv_layer* I = dynamic_cast<Conv_layer*>(&layer_i);

		GMat::Matrix tmp_Dev(this->GetNumberOfNeurons(), 1);
		tmp_Dev.ReduceMatricesToOne(*this->Derivative_Output_);
		GMat::Matrix tmp_input(this->GetNumberOfNeurons(), 1);
		tmp_input.ReduceMatricesToOne(*this->Input_);

		K->GetWeightMatrix()->Transpose();
		gradient_k.at(0) = ((*K->GetWeightMatrix()) * (*gradient_k.at(0)))->*(tmp_Dev); // [W(N,M) * delta(M,1)_K].*phi_dev(N,1) = delta(N,1)_J.
		K->GetWeightMatrix()->Transpose();

		GMat::Matrix tmp = tmp_input->*(*gradient_k.at(0));
		*this->Delta_Weight_Matrix_ += (tmp.RowSum(this->Sheet_)) * this->Learning_Rate_; // mu * (400x1 input) .* (400x1 gradient_j) then each 25 element are sum up. then we get vector of 16x1.
		GMat::Matrix* tmp_grad = gradient_k.front().Get();
		gradient_k.front().SetToNull();
		for (unsigned int s = 0; s < this->Sheet_; s++)
		{
			GMat::Matrix frame(this->Row_ * 2, this->Col_ * 2, Zeroes);
			for (unsigned int r = 0; r < this->Row_; r++)
			{
				for (unsigned int c = 0; c < this->Col_; c++)
				{
					for (int i = 0; i < 2; i++)
					{
						for (int j = 0; j < 2; j++)
						{
							frame((2 * r) + i, (2 * c) + j) = (*tmp_grad)(s * this->Row_ * this->Col_ + r * this->Col_ + c, 0) * (*this->Weight_Matrix_)(s, 0);
						}
					}
				}
			}
			gradient_k.push_back(frame);
		}
		tmp_grad->~Matrix();
	}
	else if (layer_i.LayerClassification().compare("ReLU layer") == 0 && layer_k.LayerClassification().compare("Conv layer") == 0)
	{

	}
	else // conv | pool | conv
	{

		Conv_layer* K = dynamic_cast<Conv_layer*>(&layer_k);
		Conv_layer* I = dynamic_cast<Conv_layer*>(&layer_i);

		std::vector<std::vector<int>*>* tmp_idx_table = K->GetFMTableIdx();
		GMat::shared_vector* tmp_filters = K->GetFilters();
		GMat::unique_vector tmp_grad(this->Row_, this->Col_, this->GetNumberOfSheets(), Zeroes);

		for (unsigned int f = 0; f < K->GetNumberOfSheets(); f++)
		{
			for (unsigned int s = 0; s < (unsigned int)tmp_idx_table->at(f)->size(); s++)
			{
				*tmp_grad.at(tmp_idx_table->at(f)->at(s)) += *gradient_k.at(tmp_idx_table->at(f)->at(s));
			}
		}

		for (unsigned int f = 0; f < this->GetNumberOfSheets(); f++)
		{
			*tmp_grad.at(f) = (*tmp_grad.at(f))->*(*this->Derivative_Output_->at(f));
			(*this->Delta_Weight_Matrix_)(f, 0) += ((*tmp_grad.at(f))->*(*this->Input_->at(f))).MatrixSumAllElements() * this->Learning_Rate_;
		}

		for (unsigned int s = 0; s < this->Sheet_; s++)
		{
			GMat::Matrix frame(this->Row_ * 2, this->Col_ * 2, Zeroes);
			for (unsigned int r = 0; r < this->Row_; r++)
			{
				for (unsigned int c = 0; c < this->Col_; c++)
				{
					for (int i = 0; i < 2; i++)
					{
						for (int j = 0; j < 2; j++)
						{
							frame((2 * r) + i, (2 * c) + j) = (*tmp_grad.at(s))(r, c) * (*this->Weight_Matrix_)(s, 0);
						}
					}
				}
			}
			gradient_k.at(s) = frame;
		}

		gradient_k = tmp_grad;
	}
}