#ifndef _MATRIX_H_
#define _MATRIX_H_

#include "IncluderLib.h"
#include "ExceptionHandler.h"


namespace GMat {
	class shared_vector;

	class Matrix
	{
	public:

		/*Matrix custom constructor.
		* @params:
		*
		*
		* @note:
		*/
		Matrix(unsigned int Row, unsigned int Col, ElementValues Chosen = Random_Val);

		/*Copy constructor.
		* @params: rhs - Matrix object.
		*
		* @note: Create new matrix and copy all of the data from rhs.
		*/
		Matrix(const Matrix& rhs);

		/*default destructor.*/
		~Matrix();

		/*GetIndentity
		* @params: None.
		*
		* @note: return the identity of the object. (matrix or a vector).
		*/
		Identity_ GetIdentity() const;

		/*GetDevice.
		* @param: None.
		*
		* @info: to get the best performance for each of the operations, the function return which device to use.
		*/
		Device_ GetDevice() const;

		/*GetRowSize.
		* @param: None.
		*
		* @info: return the row size of the matrix.
		*/
		unsigned int GetRowSize() const;

		/*GetColSize.
		* @param: None.
		*
		* @info: return the col size of the matrix.
		*/
		unsigned int GetColSize() const;

		/*GetMatrix.
		* @param: None.
		*
		* @info: return pointer to the array.
		*/
		float* GetMatrix() const;

		/*IsEmpty.
		* @param: None.
		*
		* @info: return true if the matrix is empty, otherwise false.
		*/
		bool IsEmpty() const;

		/*operator++.
		* @param: None.
		*
		* @info: increase each of the element in the array by one
		*/
		void operator++();

		/*operator--.
		* @param: None.
		*
		* @info: dicrease each of the element in the array by one
		*/
		void operator--();

		/*operator==.
		* @param: rhs - right hand sided matrix.
		*
		* @info: compare if the matrices are the same in size and valuse and return true if they are the same.
		*/
		bool operator==(const Matrix& rhs) const;

		/*operator!=.
		* @param: rhs - right hand sided matrix.
		*
		* @info: compare if the matrices are not the same in size or/and valuse and return true if they are not the same.
		*/
		bool operator!=(const Matrix& rhs) const;

		/*operator+.
		* @param: rhs - right hand sided matrix.
		*
		* @info: sum each element of both matrices and return new matrix.
		*/
		Matrix operator+(const Matrix& rhs);

		/*operator+.
		* @param: val - float value.
		*
		* @info: add val to each of the elements in the array and return new matrix.
		*/
		Matrix operator+(float Val);

		/*operator-.
		* @param: rhs - right hand sided matrix.
		*
		* @info: subtraction each element of both matrices and return new matrix.
		*/
		Matrix operator-(const Matrix& rhs);

		/*operator-.
		* @param: val - float value.
		*
		* @info: substract val from each of the elements in the array and return new matrix.
		*/
		Matrix operator-(float Val);

		/*operator*.
		* @param: rhs - right hand sided matrix.
		*
		* @info: return new matrix of the matrices multipication.
		*/
		Matrix operator*(const Matrix& rhs);

		/*operator*.
		* @param: Val - float value.
		*
		* @info: each element multiplied with val and kept in new matrix, then return the new matrix.
		*/
		Matrix operator*(float Val);

		/*operator^.
		* @param: ToThePowerOf - float power value.
		*
		* @info: power each of the element in the matrix by ToThePowerOf and kept the product in new matrix, then return the new matrix.
		*/
		Matrix operator^(float ToThePowerOf);

		/*operator/.
		* @param: Val - float value.
		*
		* @info: each element divided with val and kept in new matrix, then return the new matrix.
		*/
		Matrix operator/(float Val);

		/*operator->*.
		* @param: rhs - right hand sided matrix.
		*
		* @info: this operation is the same as .* operation between matrices in matlab.
		*/
		Matrix operator->*(const Matrix& rhs);

		/*operator=.
		* @param: rhs - right hand sided matrix.
		*
		* @info: erase all of the data from the current matrix and initialze it with rhs matrix, then
		*		 return a reference to the object.
		*/
		Matrix& operator=(const Matrix& rhs);

		/*operator+=.
		* @param: rhs - right hand sided matrix.
		*
		* @info: sum up the matricies, then return a reference to the object.
		*/
		Matrix& operator+=(const Matrix& rhs);

		/*operator-=.
		* @param: rhs - right hand sided matrix.
		*
		* @info: subtract the matricies, then return a reference to the object.
		*/
		Matrix& operator-=(const Matrix& rhs);

		/*operator*.
		* @param: rhs - right hand sided matrix.
		*
		* @info: return reference to the object of the matrices multipication.
		*/
		Matrix& operator*=(const Matrix& rhs);

		/*operator*.
		* @param: Val - float value.
		*
		* @info: return reference to the object of the matrix multiplied by val.
		*/
		Matrix& operator*=(float Val);
		Matrix& operator+=(float Val);
		Matrix& operator-=(float Val);
		Matrix& operator^=(float ToThePowerOf);
		Matrix& operator/=(float Val);

		friend std::ostream& operator<<(std::ostream& os, const Matrix& rhs);
		friend std::istream& operator>>(std::istream& is, Matrix& rhs);

		float& operator()(unsigned int RowIdx, unsigned int ColIdx);
		float operator()(unsigned int RowIdx, unsigned int ColIdx) const;

		void SetMatrixToZero();
		void Transpose();
		void ResizeMatrix(unsigned int Row, unsigned int Col);
		void ShafelValues();
		void PrintMatrix() const;
		void Convolution(const Matrix& Filter, shared_vector& img, Padding_ pading = WithoutZP);
		void BPConvolution(Matrix& Filter, const Matrix& GradientMat);
		void ReLUMatrix(const Matrix& source, float bias = 1);
		void SoftMaxMatrix();
		void SVMPoolMatrix(const Matrix& source);
		void AVGPoolMatrix(const Matrix& source);
		void ActivateMatrix(const Matrix& source, unsigned int AFidx, float bias);
		void DerivativeMatrix(const Matrix& source, unsigned int AFidx, float bias);
		void DotMatrices(const Matrix& rhs);
		void ReduceMatricesToOne(shared_vector& rhs);
		void Norm();
		float FindMaxVal() const;
		Matrix ColumnSum() const;
		Matrix RowSum(unsigned int SheetSize) const;
		void MatrixRotation_180();
		float MatrixSumAllElements() const;
		//Matrix Inverse();
		//float Determinant() const;
		//Matrix Eigenvalues() const;
		//Matrix* GaussianElimination(bool isInverse = false);

		void SaveMatrix(const std::string& _file) const;
		void LoadMatrix(const std::string& _file);

		//only for vector matrices.
		void PushBack(float Val);
		void PopBack();

		Device_ Optimizer(bool SelfOperations, Identity_ idt = IsAMatrix, bool Selfarithmetic = false, bool ConvOp = false) const;
		void SwapRows(int row1, int row2);

	protected:

		Device_ Chosen_Device_;
		Identity_ Idt_;
		unsigned int Row_;
		unsigned int Col_;
		float* Array_CPU_;
	};

	Matrix Log(const Matrix& rhs);
	Matrix ReLU(const Matrix& rhs);
	Matrix Tanh(const Matrix& rhs);
	Matrix SoftMaxMatrix(const Matrix& rhs);
	Matrix AvgMatrix(const Matrix& rhs);

	class MyUniquePtr
	{
	public:

		MyUniquePtr()
		{
			this->pData_ = nullptr;
			this->Is_Empty_ = true;
		}

		MyUniquePtr(const Matrix& other)
		{
			Matrix* ptr = new Matrix(other);
			this->pData_ = ptr;
			this->Is_Empty_ = false;
		}

		MyUniquePtr(MyUniquePtr& other)
		{
			if (other.Is_Point_To_Null())
			{
				this->pData_ = nullptr;
				this->Is_Empty_ = true;
			}
			else
			{
				Matrix* tmp = new Matrix(*other.Get());
				this->pData_ = tmp;
				this->Is_Empty_ = false;
				other.Release();
			}
		}

		MyUniquePtr(MyUniquePtr* other)
		{
			if (other->Is_Point_To_Null())
			{
				this->pData_ = nullptr;
				this->Is_Empty_ = true;
			}
			else
			{
				Matrix* tmp = new Matrix(*other->Get());
				this->pData_ = tmp;
				this->Is_Empty_ = false;
				other->Release();
			}
		}

		~MyUniquePtr()
		{
			this->Release();
		}

		bool Is_Point_To_Null() const
		{
			return this->Is_Empty_;
		}

		MyUniquePtr& operator=(MyUniquePtr& other)
		{
			if (other.Is_Point_To_Null())
				throw exceptionh::ExceptionHandler("Nullptr assignment is forbbiden.");

			if (this->Is_Empty_ == true || this != &other)
			{
				if (this->Is_Empty_ == false)
				{
					this->Release();
				}
				Matrix* ptr = new Matrix(*other.Get());
				this->pData_ = ptr;
				this->Is_Empty_ = false;
				other.Release();
			}

			return *this;
		}

		MyUniquePtr& operator=(MyUniquePtr* other)
		{
			if (other->Is_Point_To_Null())
				throw exceptionh::ExceptionHandler("Nullptr assignment is forbbiden.");

			if (this->Is_Empty_ == true || this != other)
			{
				if (this->Is_Empty_ == false)
				{
					this->Release();
				}
				Matrix* ptr = new Matrix(*other->Get());
				this->pData_ = ptr;
				this->Is_Empty_ = false;
				other->Release();
			}

			return *this;
		}

		MyUniquePtr& operator=(const Matrix& other)
		{
			if (this->Is_Empty_ == true || *this->pData_ != other)
			{
				if (this->Is_Empty_ == false)
				{
					delete this->pData_;
				}
				Matrix* ptr = new Matrix(other);
				this->pData_ = ptr;
				this->Is_Empty_ = false;
			}

			return *this;
		}

		Matrix& operator*()
		{
			return *this->pData_;
		}

		Matrix* operator->()
		{
			return this->pData_;
		}

		Matrix* Get()
		{
			return this->pData_;
		}

		Matrix* Get() const
		{
			return this->pData_;
		}

		void Release()
		{
			if (this->Is_Empty_ == false)
			{
				this->Is_Empty_ = true;
				delete this->pData_;
				this->pData_ = nullptr;
			}
		}

		void SetToNull()
		{
			this->pData_ = nullptr;
			this->Is_Empty_ = true;
		}

	protected:
		Matrix * pData_;
		bool Is_Empty_; //false - if the pointer point to an exist matrix, true - otherwise.
	};


	class RC
	{
	public:

		RC()
		{
			this->count_ = 0;
		}

		~RC()
		{
			this->count_ = 0;
		}

		void AddRef()
		{
			this->count_++;
		}

		int Release()
		{
			return --this->count_;
		}

		int GetCount()
		{
			return this->count_;
		}

		void SetCount(int Val)
		{
			this->count_ = Val;
		}

	protected:
		int count_;
	};


	class MySharedPtr
	{
	public:

		MySharedPtr()
		{
			this->pData_ = nullptr;
			this->Reference_ = nullptr;
			this->Is_Empty_ = true;
		}

		MySharedPtr(const Matrix& other)
		{
			Matrix* ptr = new Matrix(other);
			this->pData_ = ptr;
			this->Is_Empty_ = false;
			this->Reference_ = new RC();
			this->Reference_->AddRef();
		}

		MySharedPtr(MySharedPtr& ptr)
		{
			if (ptr.Is_Point_To_Null())
			{
				this->pData_ = nullptr;
				this->Reference_ = nullptr;
				this->Is_Empty_ = true;
			}
			else
			{
				this->pData_ = ptr.GetData();
				this->Is_Empty_ = false;
				ptr.GetRef()->AddRef();
				this->Reference_ = ptr.GetRef();
			}
		}

		MySharedPtr(MySharedPtr* ptr)
		{
			if (ptr->Is_Point_To_Null())
			{
				this->pData_ = nullptr;
				this->Reference_ = nullptr;
				this->Is_Empty_ = true;
			}
			else
			{
				this->pData_ = ptr->GetData();
				this->Is_Empty_ = false;
				ptr->GetRef()->AddRef();
				this->Reference_ = ptr->GetRef();
			}
		}

		~MySharedPtr()
		{
			if (this->Reference_->Release() == 0)
			{
				this->pData_->~Matrix();
				delete this->Reference_;
			}
			else
			{
				this->pData_ = nullptr;
				this->Reference_->Release();
				this->Reference_ = nullptr;
			}
		}

		MySharedPtr& operator=(MySharedPtr& ptr)
		{
			if (ptr.Is_Point_To_Null())
				throw exceptionh::ExceptionHandler("Nullptr assignment is forbbiden.");

			if (this->Is_Empty_ == true || this != &ptr)
			{
				if (this->Is_Empty_ == false)
				{
					this->Release();
				}
				this->pData_ = ptr.GetData();
				this->Is_Empty_ = false;
				ptr.GetRef()->AddRef();
				this->Reference_ = ptr.GetRef();
			}

			return *this;
		}

		MySharedPtr& operator=(MySharedPtr* ptr)
		{
			if (ptr->Is_Point_To_Null())
				throw exceptionh::ExceptionHandler("Nullptr assignment is forbbiden.");

			if (this->Is_Empty_ == true || this != ptr)
			{
				if (this->Is_Empty_ == false)
				{
					this->Release();
				}
				this->pData_ = ptr->GetData();
				this->Is_Empty_ = false;
				ptr->GetRef()->AddRef();
				this->Reference_ = ptr->GetRef();
			}

			return *this;
		}

		MySharedPtr& operator=(const GMat::Matrix& other)
		{
			if (this->Is_Empty_ == true || *this->pData_ != other)
			{
				if (this->Is_Empty_ == false)
				{
					this->Release();
				}
				Matrix* ptr = new Matrix(other);
				this->pData_ = ptr;
				this->Is_Empty_ = false;
				this->Reference_ = new RC();
				this->Reference_->AddRef();
			}

			return *this;
		}

		Matrix& operator*()
		{
			return *this->pData_;
		}

		Matrix* operator->()
		{
			return this->pData_;
		}

		Matrix* GetData()
		{
			return this->pData_;
		}

		RC* GetRef()
		{
			return this->Reference_;
		}

		bool Is_Point_To_Null() const
		{
			return this->Is_Empty_;
		}

		void Release()
		{
			if (this->Is_Empty_ == false)
			{
				this->Is_Empty_ = true;
				if (this->Reference_->GetCount() == 1)
				{
					delete this->Reference_;
					delete this->pData_;
				}
				else
				{
					this->Reference_->Release();
				}
				this->Reference_ = nullptr;
				this->pData_ = nullptr;
			}
		}

	protected:
		Matrix * pData_;
		RC* Reference_; //reference counter 
		bool Is_Empty_;
	};

	class shared_vector
	{
	public:
		shared_vector(unsigned int row, unsigned int col, unsigned int size, ElementValues Random_Val) :
			Size_(size)
		{
			this->Vec_ = new std::vector<MySharedPtr*>;
			for (unsigned int i = 0; i < size; i++)
			{
				Matrix tmp(row, col, Random_Val);
				MySharedPtr* tmp_ptr = new MySharedPtr(tmp);
				this->Vec_->push_back(tmp_ptr);
			}
		}
		shared_vector(shared_vector& rhs)
		{
			this->Size_ = rhs.GetSize();

			this->Vec_ = new std::vector<MySharedPtr*>;
			for (unsigned int i = 0; i < this->Size_; i++)
			{
				MySharedPtr* tmp_ptr = new MySharedPtr(rhs.at(i));
				this->Vec_->push_back(tmp_ptr);
			}
		}
		shared_vector(unsigned int size = 0) :
			Size_(size)
		{
			this->Vec_ = new std::vector<MySharedPtr*>;
			for (unsigned int i = 0; i < this->Size_; i++)
			{
				MySharedPtr* tmp_ptr = new MySharedPtr();
				this->Vec_->push_back(tmp_ptr);
			}

		}
		~shared_vector()
		{
			this->Release_Vec();
			delete this->Vec_;
		}

		shared_vector& operator=(shared_vector& rhs)
		{
			this->Release_Vec();
			this->Size_ = rhs.GetSize();
			for (unsigned int i = 0; i < this->Size_; i++)
			{
				MySharedPtr* tmp = new MySharedPtr(rhs.at(i));
				this->Vec_->push_back(tmp);
			}

			return *this;
		}
		MySharedPtr& at(int index)
		{
			return *this->Vec_->at(index);
		}
		MySharedPtr& back()
		{
			return *this->Vec_->back();
		}
		MySharedPtr& front()
		{
			return *this->Vec_->front();
		}
		void pop_back()
		{
			if (this->Vec_->size() == 0)
				throw exceptionh::ExceptionHandler("");

			delete this->Vec_->back();
			this->Vec_->pop_back();
			this->Size_ = this->Vec_->size();
		}
		void push_back(MySharedPtr* ptr)
		{
			this->Vec_->push_back(ptr);
			this->Size_ = (unsigned int)this->Vec_->size();
		}
		void push_back(MySharedPtr& ptr)
		{
			MySharedPtr* tmp_ptr = new MySharedPtr(ptr);
			this->Vec_->push_back(tmp_ptr);
			this->Size_ = (unsigned int)this->Vec_->size();
		}
		void push_back(const GMat::Matrix& rhs)
		{
			if (this->Vec_->size() == 0)
			{
				MySharedPtr* tmp_ptr = new MySharedPtr(rhs);
				this->Vec_->push_back(tmp_ptr);
			}

			else
			{
				bool Activated = false;
				for (unsigned int i = 0; i < this->Vec_->size(); i++)
				{
					if (this->Vec_->at(i)->Is_Point_To_Null())
					{
						*this->Vec_->at(i) = rhs;
						Activated = true;
						break;
					}
				}

				if (Activated == false)
				{
					MySharedPtr* tmp_ptr = new MySharedPtr(rhs);
					this->Vec_->push_back(tmp_ptr);
				}
			}
			this->Size_ = (unsigned int)this->Vec_->size();
		}
		void RangeErase(int From, int Untill)
		{
			auto it = this->Vec_->begin();
			this->Vec_->erase(it + From, it + Untill);
			this->Size_ = this->Vec_->size();
		}
		void ChangeSize(int Size)
		{
			//TODO.
		}
		void Release_Vec()
		{
			if (this->Vec_->empty() != true)
			{
				for (unsigned int i = 0; i < this->Size_; i++)
				{
					delete this->Vec_->at(i);
				}
				this->Vec_->clear();
				this->Size_ = (unsigned int)this->Vec_->size();
			}
		}


		unsigned int GetSize() const
		{
			return this->Size_;
		}

	protected:

		unsigned int Size_;
		std::vector<MySharedPtr*>* Vec_;
	};

	class unique_vector
	{
	public:
		unique_vector(unsigned int row, unsigned int col, unsigned int size, ElementValues Random_Val) :
			Size_(size)
		{
			this->Vec_ = new std::vector<MyUniquePtr*>;
			for (unsigned int i = 0; i < size; i++)
			{
				Matrix tmp(row, col, Random_Val);
				MyUniquePtr* tmp_ptr = new MyUniquePtr(tmp);
				this->Vec_->push_back(tmp_ptr);
			}
		}
		unique_vector(unique_vector& rhs)
		{
			this->Size_ = rhs.GetSize();

			this->Vec_ = new std::vector<MyUniquePtr*>;
			for (unsigned int i = 0; i < this->Size_; i++)
			{
				MyUniquePtr* tmp_ptr = new MyUniquePtr(rhs.at(i));
				this->Vec_->push_back(tmp_ptr);
			}
		}
		unique_vector(unsigned int size = 0) :
			Size_(size)
		{
			this->Vec_ = new std::vector<MyUniquePtr*>;
			for (unsigned int i = 0; i < this->Size_; i++)
			{
				MyUniquePtr* tmp_ptr = new MyUniquePtr();
				this->Vec_->push_back(tmp_ptr);
			}

		}
		~unique_vector()
		{
			this->Release_Vec();
			delete this->Vec_;
		}

		unique_vector& operator=(unique_vector& rhs)
		{
			this->Release_Vec();
			this->Size_ = rhs.GetSize();
			for (unsigned int i = 0; i < this->Size_; i++)
			{
				MyUniquePtr* tmp = new MyUniquePtr(rhs.at(i));
				this->Vec_->push_back(tmp);
			}

			return *this;
		}
		MyUniquePtr& at(int index)
		{
			return *this->Vec_->at(index);
		}
		MyUniquePtr& back()
		{
			return *this->Vec_->back();
		}
		MyUniquePtr& front()
		{
			return *this->Vec_->front();
		}
		void pop_back()
		{
			if (this->Vec_->size() == 0)
				throw exceptionh::ExceptionHandler("");

			delete this->Vec_->back();
			this->Vec_->pop_back();
			this->Size_ = this->Vec_->size();
		}
		void push_back(MyUniquePtr* ptr)
		{
			this->Vec_->push_back(ptr);
			this->Size_ = (unsigned int)this->Vec_->size();
		}
		void push_back(MyUniquePtr& ptr)
		{
			MyUniquePtr* tmp_ptr = new MyUniquePtr(ptr);
			this->Vec_->push_back(tmp_ptr);
			this->Size_ = (unsigned int)this->Vec_->size();
		}
		void push_back(const Matrix& rhs)
		{
			if (this->Vec_->size() == 0)
			{
				MyUniquePtr* tmp_ptr = new MyUniquePtr(rhs);
				this->Vec_->push_back(tmp_ptr);
			}

			else
			{
				bool Activated = false;
				for (unsigned int i = 0; i < this->Vec_->size(); i++)
				{
					if (this->Vec_->at(i)->Is_Point_To_Null())
					{
						*this->Vec_->at(i) = rhs;
						Activated = true;
						break;
					}
				}

				if (Activated == false)
				{
					MyUniquePtr* tmp_ptr = new MyUniquePtr(rhs);
					this->Vec_->push_back(tmp_ptr);
				}
			}
			this->Size_ = (unsigned int)this->Vec_->size();
		}
		void RangeErase(int From, int Untill)
		{
			auto it = this->Vec_->begin();
			this->Vec_->erase(it + From, it + Untill);
			this->Size_ = this->Vec_->size();
		}
		void ChangeSize(int Size)
		{
			// TODO
		}
		void Release_Vec()
		{
			if (this->Vec_->empty() != true)
			{
				for (unsigned int i = 0; i < this->Size_; i++)
				{
					delete this->Vec_->at(i);
				}
				this->Vec_->clear();
				this->Size_ = (unsigned int)this->Vec_->size();
			}
		}


		unsigned int GetSize() const
		{
			return this->Size_;
		}

	protected:

		unsigned int Size_;
		std::vector<MyUniquePtr*>* Vec_;
	};

	int ReverseInt(int i);

	void read_Mnist(const std::string& filename, shared_vector& vec, int Number_of_imgs = 60001);

	void read_Mnist_Label(const std::string filename, std::vector<float>& vec, int Number_of_imgs = 60001);
}

#endif // !_MATRIX_H_
