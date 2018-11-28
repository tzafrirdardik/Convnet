#ifndef _LAYER_H_
#define _LAYER_H_

#include <Matrix.h>
#include <ExceptionHandler.h>
#include <IncluderLib.h>


class Layer
{
public:

	/*Input constructor layer.
	* @params: Row - by default 32.
	*		   Col - by default 32.
	*
	* @note: the input layer contain the image data. image resoluton 32 x 32.
	*/
	Layer(unsigned int Row = 32, unsigned int Col = 32);

	/*Conv,ReLu and Pool Constructor layers.
	* @params: Row					- row size.
	*		   Col					- col size.
	*		   layer_classification - \"Conv layer\" , \"ReLU layer\" or \"Pool layer\".
	*
	* @Convolution layer: calculating values using the values of the matrices from previous layer, with
	*					  the filters.
	*
	* @ReLU layer: takes the values of the matrices from the convolution layer, and pass them through
	*			   ReLU activation function.
	*
	* @Pool layer: decrease the number of the previous layer by half, using max pooling.
	*/
	Layer(unsigned int Row, unsigned int Col, unsigned int Sheet, const std::string& layer_classification, float learning_rate, float bias, ElementValues elem);

	~Layer();

	unsigned int GetNumberOfSheets() const;
	unsigned int GetNumberOfNeurons() const;
	GMat::shared_vector& GetOutputMatrix();
	GMat::shared_vector& GetStackOfImgs();
	void ResizeOutput(unsigned int Sheet);
	virtual void Simulate() = 0;
	std::string& LayerClassification() const;

protected:

	GMat::shared_vector*		Stack_Of_Imgs_;
	GMat::shared_vector*		Output_;
	std::string*				Layer_Classification_;
	unsigned int				Number_Of_Neurons_;
	unsigned int				Sheet_;
	unsigned int				Row_;
	unsigned int				Col_;
	float						Learning_Rate_;
	float						Bias_;
};


class Pool_layer : public Layer
{
public:

	Pool_layer();
	Pool_layer(unsigned int Row, unsigned int Col, Layer* Prev_layer, unsigned int Sheet, float learning_rate, float bias);
	~Pool_layer();

	GMat::Matrix* GetWeightMatrix() const;
	void CalculateNewWeights(int average);
	void Simulate();
	void BP(Layer& layer_i, Layer& layer_k, GMat::unique_vector& gradient_k);

protected:
	
	GMat::shared_vector*		  Input_;
	GMat::shared_vector*		  Derivative_Output_;
	GMat::Matrix*				  Weight_Matrix_;
	GMat::Matrix*				  Delta_Weight_Matrix_;
	Layer*						  Prev_layer_;
};

class ReLU_Layer :public Layer
{
public:

	ReLU_Layer(unsigned int Row, unsigned int Col, Layer* Prev_layer, unsigned int Sheet, float learning_rate, float bias);
	~ReLU_Layer();

	void Simulate();

protected:

	Layer * Prev_layer_;
};

class FC_layer : public Layer
{
public:
	FC_layer(Layer* Prev_layer, unsigned int Row_Weight_mat, unsigned int Col_Weight_mat, bool IsLastLayer, unsigned int AFidx, float learning_rate, float bias);
	~FC_layer();

	GMat::Matrix* GetDerivativeMatrix() const;
	GMat::Matrix* GetWeightMatrix() const;
	GMat::Matrix* GetDeltaWeightMatrix();

	void SetActivationFunction(unsigned int Idx);
	void SetBias(float Val);
	void ShafelWeights();
	void SetDeltaWeights(float Val, unsigned int RowIdx, unsigned int ColIdx);
	void SetDeltaWeights(const GMat::Matrix& rhs);
	void CalculateNewWeights(int average);
	void Simulate();
	void BP(Layer& layer_i, Layer& layer_k, GMat::Matrix& gradient_k);//backpropagation
	void Printlayer() const;
	void IsTraining(bool training);

protected:

	GMat::Matrix*	Derivative_Output_;
	GMat::Matrix*	Weight_Matrix_;
	GMat::Matrix*	Delta_Weight_Matrix_;
	Layer*			Prev_layer_;
	unsigned int	AFidx_;
	bool			Training_;
	bool			IsLastLayer_;
};

class Input_layer :public Layer
{
public:
	Input_layer(unsigned int Row = 32, unsigned int Col = 32, WhichData_ WD = Custom_);
	~Input_layer();

	void SetImage(const std::string& file_dir_ = "");
	void Load_Mnist_Data(const std::string& file_dir, int Number_of_img = 1000);
	virtual void Simulate();

protected:

	WhichData_ WD_;
};

class Conv_layer : public Layer
{
public:
	Conv_layer(unsigned int Row, unsigned int Col, Layer* Prev_layer, unsigned int Sheet, unsigned int F_Row, unsigned int F_Col, unsigned int Number_Of_Filters, float learning_rate, float bias);
	~Conv_layer();

	GMat::shared_vector* GetDerivativeMatrix() const;
	std::vector<std::vector<int>*>* GetFMTableIdx() const;

	void SetFMTable();
	void CreateFilters();
	void ChangeFilter(const GMat::Matrix& NewFilter, unsigned int FilterIdx);
	void ChangeNumberOfFilters(unsigned int NewNumber);
	void ShafelFilters();
	void SetDeltaWeights(float Val, unsigned int RowIdx, unsigned int ColIdx, unsigned int Sheet);
	void CalculateNewWeights(int average);
	GMat::shared_vector* GetFilters();
	virtual void Simulate();
	void BP(Layer& layer_i, Layer& layer_k, GMat::unique_vector& gradient_k);

protected:

	unsigned int													Number_Of_Filters_;
	unsigned int													F_Row_;
	unsigned int													F_Col_;
	GMat::shared_vector*											Filter_Vec_;
	GMat::shared_vector*											Derivative_output_;
	GMat::shared_vector*											Delta_Filter_Vec_;
	Layer*															Prev_layer_;
	std::vector<GMat::shared_vector>*								FMTable_;//feature map table
	std::vector<std::vector<int>*>*									FMTable_idx_;
};

#endif // !_LAYER_H_

