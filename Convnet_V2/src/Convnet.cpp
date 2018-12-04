#include "Convnet.h"


Convnet::Convnet()
	:
	Learning_rate_(LearningRate)
{
	this->Size_ = 8;
	this->Topology_ = new std::vector<std::string>;
	this->Layer_list_ = new std::vector<Layer*>;
	this->Error_Vec_ = new GMat::Matrix(10, 1, Zeroes);
	this->Expected_Output_Vec_ = nullptr;
	this->Total_Error_In_Presentage_ = false;
	unsigned int row = 32;
	unsigned int col = 32;

	for (unsigned int i = 0; i < this->Size_; i++)
	{
		if (i == 0)
		{
			this->Topology_->push_back(std::string("Input layer"));
			this->Layer_list_->push_back(new Input_layer(row, col));
		}
		else if (i == 1)
		{
			row -= 4;
			col -= 4;
			this->Topology_->push_back(std::string("Conv layer"));
			this->Layer_list_->push_back(new Conv_layer(row, col, this->Layer_list_->back(), 6, Filter_size_R, Filter_size_C, 6, Learning_rate_, 1.0f));
		}
		else if (i == 2)
		{
			this->Topology_->push_back(std::string("Pool layer"));
			row = row / 2;
			col = col / 2;
			this->Layer_list_->push_back(new Pool_layer(row, col, this->Layer_list_->back(), 6, Learning_rate_, 1.0f));
		}
		else if (i == 3)
		{
			row -= 4;
			col -= 4;
			this->Topology_->push_back(std::string("Conv layer"));
			this->Layer_list_->push_back(new Conv_layer(row, col, this->Layer_list_->back(), 16, Filter_size_R, Filter_size_C, 16, Learning_rate_, 1.0f));
		}
		else if (i == 4)
		{
			this->Topology_->push_back(std::string("Pool layer"));
			row = row / 2;
			col = col / 2;
			this->Layer_list_->push_back(new Pool_layer(row, col, this->Layer_list_->back(), 16, Learning_rate_, 1.0f));
		}
		else if (i == 5)
		{
			this->Topology_->push_back(std::string("FC layer"));
			this->Layer_list_->push_back(new FC_layer(this->Layer_list_->back(), 120, this->Layer_list_->back()->GetNumberOfNeurons(), false, 1, Learning_rate_, 1.0f));
			dynamic_cast<FC_layer*>(this->Layer_list_->back())->IsTraining(true);
		}
		else if (i == 6)
		{
			this->Topology_->push_back(std::string("FC layer"));
			this->Layer_list_->push_back(new FC_layer(this->Layer_list_->back(), 84, 120, false, 1, Learning_rate_, 1.0f));
			dynamic_cast<FC_layer*>(this->Layer_list_->back())->IsTraining(true);
		}
		else
		{
			this->Topology_->push_back(std::string("FC layer"));
			this->Layer_list_->push_back(new FC_layer(this->Layer_list_->back(), 10, 84, true, 1, Learning_rate_, 1.0f));
		}
	}
}

Convnet::Convnet(float Learning_rate, float Bias)
	:
	Learning_rate_(Learning_rate)
{
	this->SetBias(Bias);
	this->Size_ = 0;
	this->Topology_ = new std::vector<std::string>;
	this->Layer_list_ = new std::vector<Layer*>;
	this->Error_Vec_ = new GMat::Matrix(16, 1, Zeroes);
	this->Expected_Output_Vec_ = nullptr;
}

Convnet::~Convnet()
{

}

unsigned int Convnet::GetNumberOfLayers() const
{
	return this->Size_;
}

void Convnet::Addlayer()
{

}

void Convnet::Eraselayer()
{

}

void Convnet::SetImage(WhichData_ data)
{
	try
	{
		if (data == Mnist_)
			throw exceptionh::Excep_No1("Please choose the amount of images you want to setup.");
		else if (data == Custom_)
			throw exceptionh::Excep_No2("Please insert the directory path.");
		else
			throw exceptionh::Excep_No3("Loading 5000 images for testing.");
	}

	catch (exceptionh::Excep_No1& e)
	{
		//int num = 0;
		//std::cin >> num;
		dynamic_cast<Input_layer*>(this->Layer_list_->at(0))->Load_Mnist_Data("C:/Users/gal/source/repos/Convnet_V2/Convnet_V2/src/train-images.idx3-ubyte", 20000);
		this->Data_labels_ = new std::vector<float>(20000);
		GMat::read_Mnist_Label("C:/Users/gal/source/repos/Convnet_V2/Convnet_V2/src/train-labels.idx1-ubyte",this->GetDataLabel(), 20000);
		GenerateExpectedOutput(this->GetDataLabel());
	}

	catch (exceptionh::Excep_No2& e)
	{
		std::string str;
		std::cin >> str;
		dynamic_cast<Input_layer*>(this->Layer_list_->at(0))->SetImage(str);
	}

	catch (exceptionh::Excep_No3& e)
	{
		dynamic_cast<Input_layer*>(this->Layer_list_->at(0))->Load_Mnist_Data("C:/Users/gal/source/repos/Convnet_V2/Convnet_V2/src/t10k-images.idx3-ubyte", ImageTestLimit);
		this->Data_labels_ = new std::vector<float>(ImageTestLimit);
		GMat::read_Mnist_Label("C:/Users/gal/source/repos/Convnet_V2/Convnet_V2/src/t10k-labels.idx1-ubyte", this->GetDataLabel(), ImageTestLimit);
		GenerateExpectedOutput(this->GetDataLabel());
	}
}

void Convnet::SetBias(float Val)
{
	for (unsigned int i = this->Size_ - 1; (i > 0) && (!this->Topology_->at(i).compare("FC_layer")); i--)
		dynamic_cast<FC_layer*>(this->Layer_list_->at(i))->SetBias(Val);
}

void Convnet::SetLearningRate(float Val)
{
	this->Learning_rate_ = Val;
}

void Convnet::SetActivationFunction(unsigned int Idx, unsigned int LayerIdx)
{
	dynamic_cast<FC_layer*>(this->Layer_list_->at(LayerIdx))->SetActivationFunction(Idx);
}

void Convnet::ShafelWeights()
{
	for (size_t i = 0; i < this->Topology_->size(); i++)
	{
		if (!this->Topology_->at(i).compare("Conv_layer"))
		{
			dynamic_cast<Conv_layer*>(this->Layer_list_->at(i))->ShafelFilters();
		}
		else if (!this->Topology_->at(i).compare("FC_layer"))
		{
			dynamic_cast<FC_layer*>(this->Layer_list_->at(i))->ShafelWeights();
		}
	}
}

std::vector<float>& Convnet::GetDataLabel()
{
	return *this->Data_labels_;
}

void Convnet::FeedForward()
{
	for (size_t i = 0; i < this->Topology_->size(); i++)
	{
		if (!this->Topology_->at(i).compare("Input layer"))
			this->Layer_list_->at(i)->Simulate();

		else if (!this->Topology_->at(i).compare("Conv layer"))
			this->Layer_list_->at(i)->Simulate();

		else if (!this->Topology_->at(i).compare("ReLU layer"))
			this->Layer_list_->at(i)->Simulate();

		else if (!this->Topology_->at(i).compare("Pool layer"))
			this->Layer_list_->at(i)->Simulate();

		else
			this->Layer_list_->at(i)->Simulate();
	}
}

void Convnet::BackPropagation()
{
	//==========calculate the cross entropy, error and the gradient====================================
	//GMat::Matrix y_k(*this->Expected_Output_Vec_->back());													 //expected output.
	//GMat::Matrix z_k((*dynamic_cast<FC_layer*>(this->Layer_list_->back())->GetWeightMatrix()) * (*this->Layer_list_->at(this->Layer_list_->size() - 2)->GetOutputMatrix().at(0)));//input to the last layer.
	//GMat::Matrix o_k(GMat::AvgMatrix(*this->Layer_list_->back()->GetOutputMatrix().at(0)));			 //logistic regression.
	//GMat::Matrix phi_dev(*dynamic_cast<FC_layer*>(this->Layer_list_->back())->GetDerivativeMatrix());
	//GMat::Matrix logo_kx(GMat::Log(o_k));															 //log(o_k(z)) = log(sigmoid(Wx)).
	//float m = (float)this->Error_Vec_->GetRowSize();												 //number of output units.
	
	//(*this->Error_Vec_) = (y_k->*logo_kx) / m;														 //Cross entropy cost function.
	//GMat::Matrix gradient_runner(((y_k - o_k)->*z_k)->*phi_dev);									 //delta = e * phi_dev = (y - o) * z * phi_dev.
	(*this->Error_Vec_) = *this->Expected_Output_Vec_->back() - *this->Layer_list_->back()->GetOutputMatrix().at(0);
	GMat::Matrix gradient_runner(this->Error_Vec_->operator->*(*dynamic_cast<FC_layer*>(this->Layer_list_->back())->GetDerivativeMatrix()));
																									 //==========calculate the delta weights for the last layer=========================================
	dynamic_cast<FC_layer*>(this->Layer_list_->at(this->Layer_list_->size() - 2))->GetOutputMatrix().at(0)->Transpose();
	(*dynamic_cast<FC_layer*>(this->Layer_list_->back())->GetDeltaWeightMatrix()) += (gradient_runner) * (*this->Layer_list_->at(this->Layer_list_->size() - 2)->GetOutputMatrix().at(0)) * this->Learning_rate_;
	dynamic_cast<FC_layer*>(this->Layer_list_->at(this->Layer_list_->size() - 2))->GetOutputMatrix().at(0)->Transpose();

	//==========calculate the delta weights for the rest of the neural net=============================
	GMat::unique_vector gradient_runner_vec;
	for (unsigned int FC_l = this->Size_ - 1; FC_l > 0; FC_l--)
	{
		// I < J < K
		if (this->Layer_list_->at(FC_l - 1)->LayerClassification().compare("FC layer") == 0)
		{
			dynamic_cast<FC_layer*>(this->Layer_list_->at(FC_l - 1))->BP(*this->Layer_list_->at(FC_l - 2), *this->Layer_list_->at(FC_l), gradient_runner);
			if (this->Layer_list_->at(FC_l - 2)->LayerClassification().compare("FC layer") != 0)
			{
				gradient_runner_vec.push_back(gradient_runner);
			}
		}
		else if (this->Layer_list_->at(FC_l - 1)->LayerClassification().compare("Pool layer") == 0)
		{
			dynamic_cast<Pool_layer*>(this->Layer_list_->at(FC_l - 1))->BP(*this->Layer_list_->at(FC_l - 2), *this->Layer_list_->at(FC_l), gradient_runner_vec);
		}
		else if (this->Layer_list_->at(FC_l - 1)->LayerClassification().compare("Conv layer") == 0)
		{
			dynamic_cast<Conv_layer*>(this->Layer_list_->at(FC_l - 1))->BP(*this->Layer_list_->at(FC_l - 2), *this->Layer_list_->at(FC_l), gradient_runner_vec);
		}
		else // ReLU layer
		{

		}
	}
}

void Convnet::SetChangeInWeights(int average)
{
	for (unsigned int i = this->Size_ - 1; i > 0; i--)
	{
		if (this->Layer_list_->at(i)->LayerClassification().compare("Conv layer") == 0)
			dynamic_cast<Conv_layer*>(this->Layer_list_->at(i))->CalculateNewWeights(average);

		else if (this->Layer_list_->at(i)->LayerClassification().compare("FC layer") == 0)
			dynamic_cast<FC_layer*>(this->Layer_list_->at(i))->CalculateNewWeights(average);

		else if (this->Layer_list_->at(i)->LayerClassification().compare("Pool layer") == 0)
			dynamic_cast<Pool_layer*>(this->Layer_list_->at(i))->CalculateNewWeights(average);
	}
}

void Convnet::Training(unsigned int NumberOfEpoch, unsigned int NumberOfTrainingPerEpoch, unsigned int NumberOfImgPerBP, unsigned int PrintTime, int average)
{
	for (unsigned int epoch = 0; epoch < NumberOfEpoch; epoch++)
	{
		for (unsigned int Subepoch = 0; Subepoch < NumberOfTrainingPerEpoch; Subepoch++)
		{
			for (unsigned int SubSubepoch = 0; SubSubepoch < NumberOfImgPerBP; SubSubepoch++)
			{
				this->FeedForward();
				this->BackPropagation();
				if (SubSubepoch < NumberOfImgPerBP - 1)
				{
					this->Data_labels_->pop_back();
					this->Expected_Output_Vec_->pop_back();
				}
			}

			this->SetChangeInWeights(average);
			if ((PrintTime != 0) && (Subepoch % PrintTime == 0))
			{
				this->PrintResult(this->GetDataLabel(), epoch + 1, Subepoch + 1);
				this->Data_labels_->pop_back();
				this->Expected_Output_Vec_->pop_back();
			}
			else
			{
				this->Data_labels_->pop_back();
				this->Expected_Output_Vec_->pop_back();
			}
		}
	}
}

void Convnet::Simulation(unsigned int NumOfSimulations)
{
	for (unsigned int i = 0; i < this->Size_; i++)
	{
		if ((!this->Layer_list_->at(i)->LayerClassification().compare("FC layer")))
			dynamic_cast<FC_layer*>(this->Layer_list_->at(i))->IsTraining(false);
	}

	this->SetImage(Mnist_test_);
	float counter = 0.0f;
	for (unsigned int i = 0; i < NumOfSimulations; i++)
	{
		this->FeedForward();
		float error = std::abs((*this->Expected_Output_Vec_->back().GetData() - *this->Layer_list_->back()->GetOutputMatrix().front().GetData()).MatrixSumAllElements());
		if (error > ErrorLimit)
		{
			counter += 1.0f;
		}
		this->GetDataLabel().pop_back();
	}
	counter /= ImageTestLimit;
	this->Total_Error_In_Presentage_ = (counter > ErrorLimitPresentage) ? false:true;
	for (unsigned int i = 0; i < this->Size_; i++)
	{
		if ((!this->Layer_list_->at(i)->LayerClassification().compare("FC layer")))
			dynamic_cast<FC_layer*>(this->Layer_list_->at(i))->IsTraining(true);
	}
}

void Convnet::SaveWeightsToDataFile()
{
	for (unsigned int i = (this->Size_ - 1); i > 1; i--)
	{
		this->Layer_list_->at(i)->SaveWeight();
	}
}

void Convnet::LoadWeightsFromDataFile()
{

}

void Convnet::GenerateExpectedOutput(const std::vector<float>& numbers)
{
	this->Expected_Output_Vec_ = new GMat::shared_vector(10, 1, (unsigned int)numbers.size(), Zeroes);
	for (int i = 0; i < (int)numbers.size(); i++)
	{
		switch ((int)numbers.at(i))
		{
		case 0: {(*this->Expected_Output_Vec_->at(i))(0, 0) = 1; break; }
		case 1: {(*this->Expected_Output_Vec_->at(i))(1, 0) = 1; break; }
		case 2: {(*this->Expected_Output_Vec_->at(i))(2, 0) = 1; break; }
		case 3: {(*this->Expected_Output_Vec_->at(i))(3, 0) = 1; break; }
		case 4: {(*this->Expected_Output_Vec_->at(i))(4, 0) = 1; break; }
		case 5: {(*this->Expected_Output_Vec_->at(i))(5, 0) = 1; break; }
		case 6: {(*this->Expected_Output_Vec_->at(i))(6, 0) = 1; break; }
		case 7: {(*this->Expected_Output_Vec_->at(i))(7, 0) = 1; break; }
		case 8: {(*this->Expected_Output_Vec_->at(i))(8, 0) = 1; break; }
		default: {(*this->Expected_Output_Vec_->at(i))(9, 0) = 1; break; }
		}
	}
}

bool Convnet::TotalError() const
{
	return this->Total_Error_In_Presentage_;
}

void Convnet::PrintConvnetTopology() const
{
	std::cout << "Convolutional Neural Net Work Topology:" << std::endl;
	for (size_t i = 0; i < this->Topology_->size(); i++)
		std::cout << "Layer No. " << i + 1 << ", " << this->Topology_->at(i).c_str() << "." << std::endl;
	std::cout << std::endl;
}

void Convnet::PrintResult(const std::vector<float>& number, unsigned int epoch, unsigned int Subepoch) const
{
	std::cout << "Epoch No. " << epoch << ": "  << std::endl << "Sub Epoch No. " << Subepoch << ": "<< "Expected number: " << number.back() << "." << std::endl;
	std::cout << "Expected output:  ";
	for (unsigned int i = 0; i < 10; i++)
	{
		std::cout << (*this->Expected_Output_Vec_->back())(0, i);
		if (i < 9)
			std::cout << ", ";
	}

	std::cout << std::endl << "NeuralNet result: ";
	dynamic_cast<FC_layer*>(this->Layer_list_->back())->Printlayer();
}

void Convnet::PrintWeights() const
{
	/*Not really necessary method.*/
}


