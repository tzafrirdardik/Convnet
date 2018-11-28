#include "Layer.h"


Conv_layer::Conv_layer(unsigned int Row, unsigned int Col, Layer* Prev_layer, unsigned int Sheet, unsigned int F_Row, unsigned int F_Col, unsigned int Number_Of_Filters, float learning_rate, float bias)
	:
	Layer(Row, Col, Sheet, "Conv layer", learning_rate, bias, Zeroes),
	Number_Of_Filters_(Number_Of_Filters),
	F_Row_(F_Row),
	F_Col_(F_Col)
{
	this->CreateFilters();
	this->Prev_layer_ = Prev_layer;
	this->Derivative_output_ = new GMat::shared_vector(Row, Col, Sheet, Zeroes);
	this->SetFMTable();
}

Conv_layer::~Conv_layer()
{

}

GMat::shared_vector* Conv_layer::GetDerivativeMatrix() const
{
	return this->Derivative_output_;
}

std::vector<std::vector<int>*>* Conv_layer::GetFMTableIdx() const
{
	return this->FMTable_idx_;
}

void Conv_layer::SetFMTable()
{
	this->FMTable_ = new std::vector<GMat::shared_vector>(this->Number_Of_Filters_);
	this->FMTable_idx_ = new std::vector<std::vector<int>*>(this->Number_Of_Filters_);
	if (this->Number_Of_Filters_ == 6)
	{
		for (unsigned int f = 0; f < this->Number_Of_Filters_; f++)
		{
			std::vector<int>* tmp_idx = new std::vector<int>(1);
			tmp_idx->at(0) = 1 + f;
			this->FMTable_->at(f) = this->Prev_layer_->GetOutputMatrix();
			this->FMTable_idx_->at(f) = tmp_idx;
		}
	}
	else
	{
		int m = 0;
		for (unsigned int f = 0; f < this->Number_Of_Filters_; f++)
		{
			if (f < 6)
			{
				GMat::shared_vector* tmp = new GMat::shared_vector(0, 0, 3, Zeroes);
				std::vector<int>* tmp_idx = new std::vector<int>(3);
				for (unsigned int i = 0; i < 3; i++)
				{
					tmp->at(i) = this->Prev_layer_->GetOutputMatrix().at((i + m) % 6);
					tmp_idx->at(i) = (i + m) % 6;
				}
				this->FMTable_->at(f) = *tmp;
				this->FMTable_idx_->at(f) = tmp_idx;
				m = (f == 5) ? 0 : (m + 1);
				tmp->~shared_vector();
			}
			else if (6 <= f && f < 15)
			{
				GMat::shared_vector* tmp = new GMat::shared_vector(0, 0, 4, Zeroes);
				std::vector<int>* tmp_idx = new std::vector<int>(4);
				for (unsigned int i = 0; i < 4; i++)
				{
					if (m < 6)
					{
						tmp->at(i) = this->Prev_layer_->GetOutputMatrix().at((i + m) % 6);
						tmp_idx->at(i) = (i + m) % 6;
					}
					else
					{
						int plus_one = (i > 1) ? 1 : 0;
						tmp->at(i) = this->Prev_layer_->GetOutputMatrix().at((i + m + plus_one) % 6);
						tmp_idx->at(i) = (i + m + plus_one) % 6;
					}
				}
				this->FMTable_->at(f) = *tmp;
				this->FMTable_idx_->at(f) = tmp_idx;
				m = (f == 14) ? 0 : (m + 1);
				tmp->~shared_vector();
			}
			else
			{
				GMat::shared_vector* tmp = new GMat::shared_vector(0, 0, 6, Zeroes);
				std::vector<int>* tmp_idx = new std::vector<int>(6);
				for (unsigned int i = 0; i < 6; i++)
				{
					tmp->at(i) = this->Prev_layer_->GetOutputMatrix().at(i);
					tmp_idx->at(i) = i;
				}
				this->FMTable_->at(f) = *tmp;
				this->FMTable_idx_->at(f) = tmp_idx;
			}
		}
	}
}

void Conv_layer::CreateFilters()
{
	this->Delta_Filter_Vec_ = new GMat::shared_vector(this->F_Row_, this->F_Col_, this->Number_Of_Filters_, Zeroes);
	this->Filter_Vec_ = new GMat::shared_vector(this->F_Row_, this->F_Col_, this->Number_Of_Filters_, Random_Val);
}

void Conv_layer::ChangeFilter(const GMat::Matrix& NewFilter, unsigned int FilterIdx)
{
	(*this->Filter_Vec_->at(FilterIdx)) = NewFilter;
}

void Conv_layer::ChangeNumberOfFilters(unsigned int NewNumber)
{
	this->Number_Of_Filters_ = NewNumber;
	/*TO DO - create or remove filters*/
}

void Conv_layer::ShafelFilters()
{
	for (unsigned int i = 0; i < this->Filter_Vec_->GetSize(); i++)
	{
		this->Filter_Vec_->at(i)->ShafelValues();
	}
}

void Conv_layer::SetDeltaWeights(float Val, unsigned int RowIdx, unsigned int ColIdx, unsigned int Sheet)
{
	(*this->Delta_Filter_Vec_->at(Sheet))(RowIdx, ColIdx) = Val;
}

void Conv_layer::CalculateNewWeights(int average)
{
	for (unsigned int i = 0; i < this->Filter_Vec_->GetSize(); i++)
	{
		*this->Filter_Vec_->at(i) += *this->Delta_Filter_Vec_->at(i) / (float)average;
		this->Filter_Vec_->at(i)->Norm();
		this->Delta_Filter_Vec_->at(i)->SetMatrixToZero();
	}
}

GMat::shared_vector* Conv_layer::GetFilters()
{
	return this->Filter_Vec_;
}

void Conv_layer::Simulate()
{
	for (unsigned int i = 0; i < this->Filter_Vec_->GetSize(); i++)
	{
		this->Output_->at(i)->SetMatrixToZero();
		this->Output_->at(i)->Convolution(*this->Filter_Vec_->at(i), this->FMTable_->at(i));
	}
}

void Conv_layer::BP(Layer& layer_i, Layer& layer_k, GMat::unique_vector& gradient_k)
{
	if (layer_i.LayerClassification().compare("Pool layer") == 0 && layer_k.LayerClassification().compare("FC layer") == 0)
	{

	}
	else if (layer_i.LayerClassification().compare("Pool layer") == 0 && layer_k.LayerClassification().compare("Pool layer") == 0)
	{
		Pool_layer* K = dynamic_cast<Pool_layer*>(&layer_k);
		Pool_layer* I = dynamic_cast<Pool_layer*>(&layer_i);

		for (unsigned int s = 0; s < this->Sheet_; s++)
		{
			gradient_k.at(s) = gradient_k.at(s)->operator->*(*this->Derivative_output_->at(s));
		}

		unsigned int i_r = I->GetOutputMatrix().at(0)->GetRowSize(), i_c = I->GetOutputMatrix().at(0)->GetRowSize();
		for (unsigned int f = 0; f < this->Sheet_; f++)
		{
			GMat::Matrix tmp(i_r, i_c, Zeroes);
			for (unsigned int f_idx = 0; f_idx < this->FMTable_->at(f).GetSize(); f_idx++)
			{
				tmp += (*this->FMTable_->at(f).at(f_idx));
			}

			for (unsigned int f_r = 0; f_r < this->F_Row_; f_r++)
			{
				for (unsigned int f_c = 0; f_c < this->F_Col_; f_c++)
				{
					float sum = 0.0f;
					for (unsigned int r = 0; r < this->Row_; r++)
					{
						for (unsigned int c = 0; c < this->Col_; c++)
						{
							sum += tmp(r + f_r, c + f_c) * (*gradient_k.at(f).Get())(r, c);
						}
					}
					(*this->Delta_Filter_Vec_->at(f))(f_r, f_c) = sum * this->Learning_Rate_;
				}
			}
		}

		GMat::unique_vector* tmp_grad_vec = new GMat::unique_vector();
		for (unsigned int f = 0; f < gradient_k.GetSize(); f++)
		{
			GMat::Matrix tmp(I->GetOutputMatrix().at(0)->GetRowSize(), I->GetOutputMatrix().at(0)->GetColSize(), Zeroes);
			tmp.BPConvolution(*this->Filter_Vec_->at(f), *gradient_k.at(0).Get());
			tmp_grad_vec->push_back(tmp);
		}

		gradient_k = *tmp_grad_vec;
		tmp_grad_vec->~unique_vector();
	}
	else if (layer_i.LayerClassification().compare("Pool layer") == 0 && layer_k.LayerClassification().compare("ReLU layer") == 0)
	{

	}
	else if (layer_i.LayerClassification().compare("Input layer") == 0 && layer_k.LayerClassification().compare("Pool layer") == 0)
	{
		Pool_layer* K = dynamic_cast<Pool_layer*>(&layer_k);
		Input_layer* I = dynamic_cast<Input_layer*>(&layer_i);

		for (unsigned int s = 0; s < this->Sheet_; s++)
		{
			gradient_k.at(s) = gradient_k.at(s)->operator->*(*this->Derivative_output_->at(s));
		}

		GMat::Matrix* tmp = I->GetOutputMatrix().at(0).GetData();
		for (unsigned int f = 0; f < this->Sheet_; f++)
		{
			for (unsigned int f_r = 0; f_r < this->F_Row_; f_r++)
			{
				for (unsigned int f_c = 0; f_c < this->F_Col_; f_c++)
				{
					float sum = 0.0f;
					for (unsigned int r = 0; r < this->Row_; r++)
					{
						for (unsigned int c = 0; c < this->Col_; c++)
						{
							sum += (*tmp)(r + f_r, c + f_c) * (*gradient_k.at(f).Get())(r, c);
						}
					}
					(*this->Delta_Filter_Vec_->at(f))(f_r, f_c) = sum * this->Learning_Rate_;
				}
			}
		}
	}
	else// Input | conv | ReLU
	{

	}
}