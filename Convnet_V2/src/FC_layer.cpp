#include "Layer.h"



FC_layer::FC_layer(Layer* Prev_layer, unsigned int Row_Weight_mat, unsigned int Col_Weight_mat, bool IsLastLayer, unsigned int AFidx, float learning_rate, float bias)
	:
	Layer(Row_Weight_mat, 1, 1, "FC layer", learning_rate, bias, Random_Val),
	AFidx_(AFidx),
	IsLastLayer_(IsLastLayer)
{
	this->Prev_layer_ = Prev_layer;
	this->Weight_Matrix_ = new GMat::Matrix(Row_Weight_mat, Col_Weight_mat);
	this->Delta_Weight_Matrix_ = new GMat::Matrix(Row_Weight_mat, Col_Weight_mat, Zeroes);
	this->Derivative_Output_ = new GMat::Matrix(Row_Weight_mat, 1, Ones);
	this->Derivative_Output_->SoftMaxMatrix();
}

FC_layer::~FC_layer()
{

}

GMat::Matrix* FC_layer::GetDerivativeMatrix() const
{
	return this->Derivative_Output_;
}

GMat::Matrix* FC_layer::GetWeightMatrix() const
{
	return this->Weight_Matrix_;
}

GMat::Matrix* FC_layer::GetDeltaWeightMatrix()
{
	return this->Delta_Weight_Matrix_;
}

void FC_layer::SetActivationFunction(unsigned int Idx)
{
	this->AFidx_ = Idx;
}

void FC_layer::SetBias(float Val)
{
	this->Bias_ = Val;
}

void FC_layer::ShafelWeights()
{
	this->Weight_Matrix_->ShafelValues();
}

void FC_layer::SetDeltaWeights(float Val, unsigned int RowIdx, unsigned int ColIdx)
{
	(*this->Delta_Weight_Matrix_)(RowIdx, ColIdx) = Val;
}

void FC_layer::SetDeltaWeights(const GMat::Matrix& rhs)
{
	(*this->Delta_Weight_Matrix_) += rhs;
}

void FC_layer::CalculateNewWeights(int average)
{
	(*this->Weight_Matrix_) += ((*this->Delta_Weight_Matrix_) / (float)average);
	this->Weight_Matrix_->Norm();
	this->Delta_Weight_Matrix_->SetMatrixToZero();
}

void FC_layer::Simulate()
{
	if (this->Prev_layer_->LayerClassification().compare("FC layer") != 0)//the first FC layer connect btw the image proccessing to the FC part.
	{
		GMat::Matrix input_tmp(this->Prev_layer_->GetNumberOfNeurons(), 1);
		input_tmp.ReduceMatricesToOne(this->Prev_layer_->GetOutputMatrix());
		this->Output_->at(0)->ActivateMatrix((*this->Weight_Matrix_) * (input_tmp),this->AFidx_,this->Bias_);
		this->Derivative_Output_->ActivateMatrix((*this->Weight_Matrix_) * (input_tmp), this->AFidx_, this->Bias_);
	}

	else
	{
		if (this->IsLastLayer_ == false)
		{
			this->Output_->at(0)->ActivateMatrix((*this->Weight_Matrix_) * (*this->Prev_layer_->GetOutputMatrix().at(0)), this->AFidx_, this->Bias_);
			this->Derivative_Output_->DerivativeMatrix((*this->Weight_Matrix_) * (*this->Prev_layer_->GetOutputMatrix().at(0)), this->AFidx_, this->Bias_);

			if (this->Training_ == true)
			{
				std::mt19937 gen((unsigned int)time(NULL));
				std::uniform_real_distribution<float> dist(0, 1);
				for (unsigned int i = 0; i < this->Number_Of_Neurons_; i++)
				{
					if (dist(gen) < 0.5)
					{
						(*this->Output_->at(0))(i, 0) = 0;
						(*this->Derivative_Output_)(i, 0) = 0;
					}
				}
			}
		}

		else
		{
			this->Output_->at(0)->ActivateMatrix((*this->Weight_Matrix_) * (*this->Prev_layer_->GetOutputMatrix().at(0)), this->AFidx_, this->Bias_);
			this->Derivative_Output_->DerivativeMatrix((*this->Weight_Matrix_) * (*this->Prev_layer_->GetOutputMatrix().at(0)), this->AFidx_, this->Bias_);
		}
	}
}

void FC_layer::BP(Layer& layer_i, Layer& layer_k, GMat::Matrix& gradient_k)
{
	if (layer_i.LayerClassification().compare("FC layer") == 0)
	{
		// I < J (current layer) < K
		FC_layer* K = dynamic_cast<FC_layer*>(&layer_k);
		FC_layer* I = dynamic_cast<FC_layer*>(&layer_i);

		K->GetWeightMatrix()->Transpose();
		gradient_k = ((*K->GetWeightMatrix()) * gradient_k)->*(*this->Derivative_Output_); // [W(N,M) * delta(M,1)_K].*phi_dev(N,1) = delta(N,1)_J.
		K->GetWeightMatrix()->Transpose();

		I->GetOutputMatrix().at(0)->Transpose();
		(*this->Delta_Weight_Matrix_) += (gradient_k * (*I->GetOutputMatrix().at(0))) * this->Learning_Rate_; // D_W(M,N) += delta(M,1) * y_i(1,N)
		I->GetOutputMatrix().at(0)->Transpose();
	}
	else if (layer_i.LayerClassification().compare("Pool layer") == 0)
	{
		// I < J (current layer) < K
		FC_layer* K = dynamic_cast<FC_layer*>(&layer_k);
		Pool_layer* I = dynamic_cast<Pool_layer*>(&layer_i);

		K->GetWeightMatrix()->Transpose();
		gradient_k = ((*K->GetWeightMatrix()) * gradient_k)->*(*this->Derivative_Output_); // [W(N,M) * delta(M,1)_K].*phi_dev(N,1) = delta(N,1)_J.
		K->GetWeightMatrix()->Transpose();

		GMat::Matrix input_tmp(this->Prev_layer_->GetNumberOfNeurons(), 1);
		input_tmp.ReduceMatricesToOne(this->Prev_layer_->GetOutputMatrix());
		input_tmp.Transpose();
		(*this->Delta_Weight_Matrix_) += (gradient_k * input_tmp) * this->Learning_Rate_; // D_W(M,N) += delta(M,1) * y_i(1,N)
	}
	else if (layer_i.LayerClassification().compare("Conv layer") == 0)
	{
		// I < J (current layer) < K
		FC_layer* K = dynamic_cast<FC_layer*>(&layer_k);
		Conv_layer* I = dynamic_cast<Conv_layer*>(&layer_i);

		K->GetWeightMatrix()->Transpose();
		gradient_k = ((*K->GetWeightMatrix()) * gradient_k)->*(*this->Derivative_Output_); // [W(N,M) * delta(M,1)_K].*phi_dev(N,1) = delta(N,1)_J.
		K->GetWeightMatrix()->Transpose();

		GMat::Matrix input_tmp(this->Prev_layer_->GetNumberOfNeurons(), 1);
		input_tmp.ReduceMatricesToOne(this->Prev_layer_->GetOutputMatrix());
		input_tmp.Transpose();
		(*this->Delta_Weight_Matrix_) += (gradient_k * input_tmp) * this->Learning_Rate_; // D_W(M,N) += delta(M,1) * y_i(1,N)
	}
	else//ReLU layer.
	{
		// I < J (current layer) < K
		FC_layer* K = dynamic_cast<FC_layer*>(&layer_k);
		ReLU_Layer* I = dynamic_cast<ReLU_Layer*>(&layer_i);

		K->GetWeightMatrix()->Transpose();
		gradient_k = ((*K->GetWeightMatrix()) * gradient_k)->*(*this->Derivative_Output_); // [W(N,M) * delta(M,1)_K].*phi_dev(N,1) = delta(N,1)_J.
		K->GetWeightMatrix()->Transpose();

		GMat::Matrix input_tmp(this->Prev_layer_->GetNumberOfNeurons(), 1);
		input_tmp.ReduceMatricesToOne(this->Prev_layer_->GetOutputMatrix());
		input_tmp.Transpose();
		(*this->Delta_Weight_Matrix_) += (gradient_k * input_tmp) * this->Learning_Rate_; // D_W(M,N) += delta(M,1) * y_i(1,N)
	}
}

void FC_layer::Printlayer() const
{
	for (unsigned int i = 0; i < 10; i++)
	{
		std::cout << (*this->Output_->at(0))(0, i);
		if (i < 9)
			std::cout << ", ";
	}
	PRINT_EMPTY_LINE
}

void FC_layer::IsTraining(bool training)
{
	this->Training_ = training;
}

void FC_layer::SaveWeight()
{
	unsigned int r_size = this->Weight_Matrix_->GetRowSize();
	unsigned int c_size = this->Weight_Matrix_->GetColSize();
	std::string tmp(STR(FC_layer_weights_size_));
	tmp.append(STR(r_size));
	tmp.append(STR(x));
	tmp.append(STR(c_size));
	tmp.append(STR(.txt));
	this->Weight_Matrix_->SaveMatrix(tmp);
}