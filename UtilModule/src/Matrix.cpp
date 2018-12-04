#include "Matrix.h"


namespace GMat {
	Matrix::Matrix(unsigned int Row, unsigned int Col, ElementValues Chosen)
		:
		Row_(Row),
		Col_(Col)
	{
		this->Array_CPU_ = new float[Row * Col];
		if (Chosen != DontCare)
		{
			std::mt19937 gen((unsigned int)time(NULL));
			std::uniform_real_distribution<float> dist(-1,1);
			for (unsigned int r = 0; r < this->Row_;r++)
			{
				for (unsigned int c = 0; c < this->Col_; c++)
				{
					if (Chosen == Zeroes)
						this->Array_CPU_[r * Col_ + c] = 0.0;
					else if (Chosen == Ones)
						this->Array_CPU_[r * Col_ + c] = 1;
					else
						this->Array_CPU_[r * Col_ + c] = dist(gen);
				}
			}
		}
		this->Idt_ = ((Row == 1) || (Col == 1)) ? IsAVector : IsAMatrix;
		this->Chosen_Device_ = this->Optimizer(true);
	}

	Matrix::Matrix(const Matrix& rhs)
	{
		this->Row_ = rhs.GetRowSize();
		this->Col_ = rhs.GetColSize();
		this->Idt_ = rhs.GetIdentity();
		this->Chosen_Device_ = rhs.GetDevice();
		this->Array_CPU_ = new float[this->Row_ * this->Col_];
		float* tmp = rhs.GetMatrix();

		for (unsigned int r = 0; r < this->Row_; r++)
		{
			for (unsigned int c = 0; c < this->Col_; c++)
			{
				this->Array_CPU_[r * Col_ + c] = tmp[r * Col_ + c];
			}
		}
	}

	Matrix::~Matrix()
	{
		if (this->Array_CPU_ != NULL)
			delete this->Array_CPU_;
	}

	Identity_ Matrix::GetIdentity() const
	{
		return this->Idt_;
	}

	Device_ Matrix::GetDevice() const
	{
		return this->Chosen_Device_;
	}

	unsigned int Matrix::GetRowSize() const
	{
		return this->Row_;
	}

	unsigned int Matrix::GetColSize() const
	{
		return this->Col_;
	}

	float* Matrix::GetMatrix() const
	{
		return this->Array_CPU_;
	}

	bool Matrix::IsEmpty() const
	{
		if (this->Row_ == 0 || this->Col_ == 0)
			return true;
		return false;
	}

	void Matrix::operator++()
	{
		this->operator+=(1);
	}

	void Matrix::operator--()
	{
		this->operator-=(1);
	}

	bool Matrix::operator==(const Matrix& rhs) const
	{
		if (this->Row_ != rhs.GetRowSize() || this->Col_ != rhs.GetColSize())
			return false;
		if (this->Array_CPU_ == rhs.GetMatrix())
			return true;

		float* tmp = rhs.GetMatrix();
		if (tmp == NULL)
			return false;

		for (unsigned int r = 0; r < this->Row_; r++)
			for (unsigned int c = 0; c < this->Col_; c++)
				if (this->Array_CPU_[r * this->Col_ + c] != tmp[r * this->Col_ + c])
					return false;

		return true;
	}

	bool Matrix::operator!=(const Matrix& rhs) const
	{
		return !this->operator==(rhs);
	}

	Matrix Matrix::operator+(const Matrix& rhs)
	{
		Matrix tmp(*this);
		tmp += rhs;
		return tmp;
	}

	Matrix Matrix::operator+(float Val)
	{
		Matrix tmp(*this);
		float* tmp_arr = tmp.GetMatrix();
		if (this->Chosen_Device_ == GPU_)
		{
			concurrency::array_view<float, 1> target(this->Row_ * this->Col_, tmp.GetMatrix());
			parallel_for_each(target.extent, [=](concurrency::index<1> idx) restrict(amp)
			{
				target[idx] += Val;
			});
			target.synchronize();
		}
		else
		{
			for (unsigned int r = 0; r < this->Row_; r++)
			{
				for (unsigned int c = 0; c < this->Col_; c++)
				{
					tmp_arr[r * this->Col_ + c] += Val;
				}
			}
		}
		return tmp;
	}

	Matrix Matrix::operator-(const Matrix& rhs)
	{
		Matrix tmp(*this);
		tmp -= rhs;
		return tmp;
	}

	Matrix Matrix::operator-(float Val)
	{
		return this->operator+(-Val);
	}

	Matrix Matrix::operator*(const Matrix& rhs)
	{
		Matrix tmp(*this);
		tmp *= rhs;
		return tmp;
	}

	Matrix Matrix::operator*(float Val)
	{
		Matrix tmp(*this);
		tmp *= Val;
		return tmp;
	}

	Matrix Matrix::operator^(float ToThePowerOf)
	{
		Matrix tmp(*this);
		tmp ^= ToThePowerOf;
		return tmp;
	}

	Matrix Matrix::operator/(float Val)
	{
		if (Val == 0)
			throw exceptionh::ExceptionHandler("You cannot divide by zero.");

		Matrix tmp(*this);
		tmp /= Val;
		return tmp;
	}

	Matrix Matrix::operator->*(const Matrix& rhs)
	{
		Matrix tmp(rhs);
		for (unsigned int r = 0; r < this->Row_; r++)
		{
			for (unsigned int c = 0; c < this->Col_; c++)
			{
				tmp(r, c) = this->Array_CPU_[r * this->Col_ + c] * rhs(r, c);
			}
		}
		return tmp;
	}

	Matrix& Matrix::operator=(const Matrix& rhs)
	{
		if (this != &rhs)
		{
			if (this->Array_CPU_ != NULL)
			{
				delete this->Array_CPU_;
			}
			this->Row_ = rhs.GetRowSize();
			this->Col_ = rhs.GetColSize();
			this->Idt_ = rhs.GetIdentity();
			this->Chosen_Device_ = rhs.GetDevice();
			this->Array_CPU_ = new float[Row_ * Col_];
			float* tmp = rhs.GetMatrix();

			if (this->Chosen_Device_ == GPU_)
			{
				concurrency::array_view<const float, 1> other((int)(this->Row_ * this->Col_), tmp);
				concurrency::array_view<float, 1> target((int)(this->Row_ * this->Col_), this->Array_CPU_);

				target.discard_data();
				parallel_for_each(target.extent, [=](concurrency::index<1> idx) restrict(amp)
				{
					target[idx] = other[idx];
				});
				target.synchronize();
			}
			else
			{
				for (unsigned int r = 0; r < this->Row_; r++)
				{
					for (unsigned int c = 0; c < this->Col_; c++)
					{
						this->Array_CPU_[r * this->Col_ + c] = tmp[r * this->Col_ + c];
					}
				}
			}
		}

		return *this;
	}

	Matrix& Matrix::operator+=(const Matrix& rhs)
	{
		float* tmp = rhs.GetMatrix();
		if (this->Chosen_Device_ == GPU_)
		{
			concurrency::array_view<const float, 1> other((int)(this->Row_ * this->Col_), tmp);
			concurrency::array_view<float, 1> target((int)(this->Row_ * this->Col_), this->Array_CPU_);

			parallel_for_each(target.extent, [=](concurrency::index<1> idx) restrict(amp)
			{
				target[idx] += other[idx];
			});
			target.synchronize();
		}

		else
		{
			for (unsigned int r = 0; r < this->Row_; r++)
			{
				for (unsigned int c = 0; c < this->Col_; c++)
				{
					this->Array_CPU_[r * this->Col_ + c] += tmp[r * this->Col_ + c];
				}
			}
		}
		return *this;
	}

	Matrix& Matrix::operator-=(const Matrix& rhs)
	{
		float* tmp = rhs.GetMatrix();
		if (this->Chosen_Device_ == GPU_)
		{
			concurrency::array_view<const float, 1> other((int)(this->Row_ * this->Col_), tmp);
			concurrency::array_view<float, 1> target((int)(this->Row_ * this->Col_), this->Array_CPU_);

			parallel_for_each(target.extent, [=](concurrency::index<1> idx) restrict(amp)
			{
				target[idx] += other[idx];
			});
			target.synchronize();
		}

		else
		{
			for (unsigned int r = 0; r < this->Row_; r++)
			{
				for (unsigned int c = 0; c < this->Col_; c++)
				{
					this->Array_CPU_[r * this->Col_ + c] -= tmp[r * this->Col_ + c];
				}
			}
		}
		return *this;
	}

	Matrix& Matrix::operator*=(const Matrix& rhs)
	{
		float* tmp = rhs.GetMatrix();
		float* target = new float[this->Row_ * rhs.GetColSize()];

		switch (Optimizer(false, rhs.GetIdentity()))
		{
		case 0://GPU
		{
			int size = (int)this->Col_;
			concurrency::array_view<const float, 2> a((int)this->Row_, (int)this->Col_, this->Array_CPU_);
			concurrency::array_view<const float, 2> b((int)this->Col_, (int)rhs.GetColSize(), tmp);
			concurrency::array_view<float, 2> c((int)this->Row_, (int)rhs.GetColSize(), target);
			c.discard_data();

			concurrency::parallel_for_each(c.extent, [=](concurrency::index<2> idx) restrict(amp)
			{
				int row = idx[0];
				int col = idx[1];
				float sum = 0.0f;

				for (int i = 0; i < size; i++)
				{
					sum += a(row, i) * b(i, col);
				}

				c[idx] = sum;
			}
			);
			c.synchronize();
			break;
		}
		case 1://CPU
		{
			unsigned int col = rhs.GetColSize();
			for (unsigned int i = 0; i < this->Row_; i++)
			{
				for (unsigned int j = 0; j < col; j++)
				{
					float sum = 0.0f;
					for (unsigned int k = 0; k < this->Col_; k++)
					{
						sum += this->Array_CPU_[i * this->Col_ + k] * tmp[k * col + j];
					}
					target[i * col + j] = sum;
				}
			}
			break;
		}
		default: // CPU parallel
		{
			size_t size_n = size_t(this->Row_), size_m = size_t(this->Col_), size_w = size_t(rhs.GetColSize());
			float* A = this->Array_CPU_;
			concurrency::parallel_for(size_t(0), size_n, [&](size_t i)
			{
				for (size_t j = 0; j < size_w; j++)
				{
					float sum = 0.0f;
					for (size_t k = 0; k < size_m; k++)
					{
						sum += A[i * size_m + k] * tmp[k * size_w + i];
					}
					target[i * size_w + j] = sum;
				}
			});
			break;
		}
		}

		this->Col_ = rhs.GetColSize();
		delete this->Array_CPU_;
		this->Array_CPU_ = target;

		return *this;
	}

	Matrix& Matrix::operator*=(float Val)
	{
		if (this->Chosen_Device_ == GPU_)
		{
			concurrency::array_view<float, 1> target((int)(this->Row_ * this->Col_), this->Array_CPU_);
			parallel_for_each(target.extent, [=](concurrency::index<1> idx) restrict(amp)
			{
				target[idx] *= Val;
			});
			target.synchronize();
		}

		else
		{
			for (unsigned int r = 0; r < this->Row_; r++)
			{
				for (unsigned int c = 0; c < this->Col_; c++)
				{
					this->Array_CPU_[r * this->Col_ + c] *= Val;
				}
			}
		}
		return *this;
	}

	Matrix& Matrix::operator+=(float Val)
	{
		if (this->Chosen_Device_ == GPU_)
		{
			concurrency::array_view<float, 1> target((int)(this->Row_ * this->Col_), this->Array_CPU_);
			parallel_for_each(target.extent, [=](concurrency::index<1> idx) restrict(amp)
			{
				target[idx] += Val;
			});
			target.synchronize();
		}

		else
		{
			for (unsigned int r = 0; r < this->Row_; r++)
			{
				for (unsigned int c = 0; c < this->Col_; c++)
				{
					this->Array_CPU_[r * this->Col_ + c] += Val;
				}
			}
		}
		return *this;
	}

	Matrix& Matrix::operator-=(float Val)
	{
		this->operator+=(-Val);
		return *this;
	}

	Matrix& Matrix::operator^=(float ToThePowerOf)
	{
		if (this->Chosen_Device_ == GPU_)
		{
			concurrency::array_view<float, 1> target((int)(this->Row_ * this->Col_), this->Array_CPU_);
			parallel_for_each(target.extent, [=](concurrency::index<1> idx) restrict(amp)
			{
				target[idx] = concurrency::precise_math::powf(target[idx], ToThePowerOf);
			});
			target.synchronize();
		}

		else
		{
			size_t size_n = size_t(this->Row_), size_w = size_t(this->Col_);
			float* tmp = this->Array_CPU_;
			concurrency::parallel_for(size_t(0), size_n, [&](size_t i)
			{
				for (size_t j = 0; j < size_w; j++)
				{
					tmp[i * size_w + j] = std::powf(tmp[i * size_w + j], ToThePowerOf);
				}
			});
		}
		return *this;
	}

	Matrix& Matrix::operator/=(float Val)
	{
		if (Val == 0)
			throw exceptionh::ExceptionHandler("You cannot divide by zero.");

		this->operator*=(1 / Val);
		return *this;
	}

	std::ostream& operator<<(std::ostream& os, const Matrix& rhs)
	{
		float* tmp = rhs.GetMatrix();
		for (unsigned int r = 0; r < rhs.Row_; r++)
		{
			os << tmp[r * rhs.Col_];
			for (unsigned int c = 1; c < rhs.Col_; c++)
				os << ", " << tmp[r * rhs.Col_ + c];
			os << std::endl;
		}
		os << std::endl;
		return os;
	}

	std::istream& operator>>(std::istream& is, Matrix& rhs)
	{
		float* tmp = rhs.GetMatrix();
		for (unsigned int r = 0; r < rhs.Row_; r++)
			for (unsigned int c = 0; c < rhs.Col_; c++)
				is >> tmp[r * rhs.Col_ + c];
		return is;
	}

	float& Matrix::operator()(unsigned int RowIdx, unsigned int ColIdx)
	{
		return this->Array_CPU_[RowIdx * this->Col_ + ColIdx];
	}

	float Matrix::operator()(unsigned int RowIdx, unsigned int ColIdx) const
	{
		return this->Array_CPU_[RowIdx * this->Col_ + ColIdx];
	}

	void Matrix::SetMatrixToZero()
	{
		if (this->Chosen_Device_ == GPU_)
		{
			concurrency::array_view<float, 1> target((int)(this->Row_ * this->Col_), this->Array_CPU_);
			parallel_for_each(target.extent, [=](concurrency::index<1> idx) restrict(amp)
			{
				target[idx] = 0.0f;
			});
			target.synchronize();
		}

		else
		{
			for (unsigned int r = 0; r < this->Row_; r++)
			{
				for (unsigned int c = 0; c < this->Col_; c++)
				{
					this->Array_CPU_[r * this->Col_ + c] = 0.0f;
				}
			}
		}
	}

	void Matrix::Transpose()
	{
		concurrency::extent<2> e((int)this->Row_, (int)this->Col_);
		concurrency::extent<2> e2((int)this->Col_, (int)this->Row_);
		float* Arr = new float[this->Row_ * this->Col_];
		concurrency::array_view<float, 2> other(e2, Arr);
		concurrency::array_view<float, 2> target(e, this->Array_CPU_);

		if (this->Col_ >= 2)
		{
			parallel_for_each(e.tile<1, 2>(), [=](concurrency::tiled_index<1, 2> t_idx) restrict(amp)
			{
				other(t_idx.global[1], t_idx.global[0]) = target(t_idx.global[0], t_idx.global[1]);
			});
			target.synchronize();
		}
		else
		{
			parallel_for_each(e.tile<2, 1>(), [=](concurrency::tiled_index<2, 1> t_idx) restrict(amp)
			{
				other(t_idx.global[1], t_idx.global[0]) = target(t_idx.global[0], t_idx.global[1]);
			});
			target.synchronize();
		}

		unsigned int tmp = this->Row_;
		this->Row_ = this->Col_;
		this->Col_ = tmp;
		delete[] this->Array_CPU_;
		this->Array_CPU_ = Arr;
	}

	void Matrix::ResizeMatrix(unsigned int Row, unsigned int Col)
	{
		if (this->Array_CPU_ != NULL)
		{
			delete this->Array_CPU_;
		}

		this->Row_ = Row;
		this->Col_ = Col;
		this->Array_CPU_ = new float[Row * Col];
		ShafelValues();
	}

	void Matrix::ShafelValues()
	{
		std::mt19937 gen((unsigned int)time(NULL));
		std::uniform_real_distribution<float> dist(-1, 1);
		for (unsigned int r = 0; r < this->Row_; r++)
			for (unsigned int c = 0; c < this->Col_; c++)
				this->Array_CPU_[r * this->Col_ + c] = dist(gen);
	}

	void Matrix::PrintMatrix() const
	{
		PRINT(STR(Array Elements:))
		for (unsigned int r = 0; r < this->Row_; r++)
		{
			for (unsigned int c = 0; c < this->Col_; c++)
			{
				std::cout << this->Array_CPU_[r * this->Col_ + c] << ", ";
			}
			PRINT_EMPTY_LINE
		}
		PRINT_EMPTY_LINE
	}

	void Matrix::Convolution(const Matrix& Filter, shared_vector& img, Padding_ pading)
	{
		int radius = (int)Filter.GetRowSize() / 2;
		Device_ tool = img.at(0)->Optimizer(false, IsAMatrix, false, true);
		int row = (int)img.at(0)->GetRowSize(), col = (int)img.at(0)->GetColSize();
		float* result = this->Array_CPU_;
		float* filter = Filter.GetMatrix();
		int filter_size = Filter.GetRowSize();
		size_t size = (size_t)img.GetSize();
		Padding_ pad = pading;
		int lowlimit = (pad == WithoutZP) ? radius : 0;
		int upperlimit = (pad == WithoutZP) ? radius : 0;

		if (tool == CPU_parallel_)
		{
			concurrency::parallel_for(size_t(0), size, [&](size_t i)
			{
				float* tmp_img = img.at((int)i)->GetMatrix();
				for (int y = lowlimit; y < (row - upperlimit); y++)//single input
				{
					for (int x = lowlimit; x < (col - upperlimit); x++)//single input
					{
						float sum = 0.0f;
						for (int p = -radius; p <= radius; p++)//filter kernal (rows)
						{
							for (int q = -radius; q <= radius; q++)//filter kernal (cols)
							{
								if ((y + p >= 0) && (x + q >= 0) && (y + p < row) && (x + q < col))
								{
									sum += filter[(p + radius) * filter_size + (q + radius)] * tmp_img[(y + p) * col + x + q];
								}
							}
						}
						result[(y - lowlimit) * (col - 2 * lowlimit) + (x - lowlimit)] += sum;
					}
				}
			});
		}
		else // gpu
		{
			for (int f = 0; f < (int)img.GetSize(); f++)
			{
				concurrency::extent<2> e(row, col);
				concurrency::array_view<const float, 2> a((int)Filter.GetRowSize(), (int)Filter.GetColSize(), filter);
				concurrency::array_view<const float, 2> b(e, img.at(f)->GetMatrix());
				concurrency::array_view<float, 2> c((int)this->Row_, (int)this->Col_, result);

				concurrency::parallel_for_each(b.extent, [=](concurrency::index<2> idx) restrict(amp)
				{
					if (pad == WithZP)
					{
						float sum = 0.0f;
						for (int r = -radius; r <= radius; r++)
						{
							for (int c = -radius; c <= radius; c++)
							{
								if ((idx[0] + r >= 0) && (idx[1] + c >= 0) && (idx[0] + r < row) && (idx[1] + c < col))
								{
									sum += a[r + radius][c + radius] * b[idx[0] + r][idx[1] + c];
								}
							}
						}
						c[idx] = sum;
					}
					else
					{
						if ((radius <= idx[0]) && (radius <= idx[1]) && (idx[0] <= e[0] - radius) && (idx[1] <= e[1] - radius))
						{
							float sum = 0.0f;
							for (int r = -radius; r <= radius; r++)
							{
								for (int c = -radius; c <= radius; c++)
								{
									if ((idx[0] + r >= 0) && (idx[1] + c >= 0) && (idx[0] + r < row) && (idx[1] + c < col))
									{
										sum += a[r + radius][c + radius] * b[idx[0] + r][idx[1] + c];
									}
								}
							}
							c[idx[0] - radius][idx[1] - radius] = sum;
						}
					}
				});
				c.synchronize();
			}
		}
	}

	void Matrix::BPConvolution(Matrix& Filter, const Matrix& GradientMat)
	{
		Filter.MatrixRotation_180();
		int radius = Filter.GetRowSize() / 2;
		int row = GradientMat.GetRowSize(), col = GradientMat.GetColSize(), f_col = Filter.GetColSize();
		float* tmp_filter = Filter.GetMatrix();
		float* tmp_grad = GradientMat.GetMatrix();

		for (int r = -radius; r < row + radius; r++)
		{
			for (int c = -radius; c < col + radius; c++)
			{
				float sum = 0.0f;
				for (int p = -radius; p <= radius; p++)
				{
					for (int q = -radius; q <= radius; q++)
					{
						if ((0 <= r + p) && (0 <= c + q) && (r + p < row) && (c + q < col))
						{
							sum += tmp_filter[(p + radius) * f_col + (q + radius)] * tmp_grad[(r + p) * col + c + q];
						}
					}
				}
				this->Array_CPU_[(r + radius) * this->Col_ + (c + radius)] = sum;
			}
		}
		Filter.MatrixRotation_180();
	}

	void Matrix::ReLUMatrix(const Matrix& source, float bias)
	{
		unsigned int row = source.GetRowSize(), col = source.GetColSize();
		float* tmp_arr = source.GetMatrix();
		float* tmp_arr2 = this->Array_CPU_;

		if (source.Optimizer(true, source.GetIdentity(), true) == GPU_)
		{
			concurrency::extent<1> e((int)(row * col));
			concurrency::array_view<const float, 1> other(e, tmp_arr);
			concurrency::array_view<float, 1> target(e, tmp_arr2);
			concurrency::parallel_for_each(target.extent, [=](concurrency::index<1> idx) restrict(amp)
			{
				target[idx] = (other[idx] > 0) ? other[idx] : 0.0f;
			});
			target.synchronize();
		}
		else
		{
			size_t size_n = size_t(row), size_w = size_t(col);
			concurrency::parallel_for(size_t(0), size_n, [&](size_t i)
			{
				for (size_t j = 0; j < size_w; j++)
				{
					tmp_arr2[i * size_w + j] = (tmp_arr[i * size_w + j] > 0) ? tmp_arr[i * size_w + j] : 0.0f;
				}
			});
		}
	}

	void Matrix::SoftMaxMatrix()
	{
		float sum = 0.0f;
		float maxval = this->FindMaxVal();
		if (this->Chosen_Device_ == GPU_)
		{
			concurrency::array_view<float, 1> target((int)(this->Row_ * this->Col_), this->Array_CPU_);

			parallel_for_each(target.extent, [=](concurrency::index<1> idx) restrict(amp)
			{
				target[idx] = concurrency::precise_math::expf(target[idx] / maxval);
			});
			target.synchronize();
		}
		else
		{
			for (unsigned int r = 0; r < this->Row_; r++)
			{
				for (unsigned int c = 0; c < this->Col_; c++)
				{
					this->Array_CPU_[r * this->Col_ + c] = std::expf(this->Array_CPU_[r * this->Col_ + c] / maxval);
				}
			}
		}

		for (unsigned int i = 0; i < this->Row_ * this->Col_; i++)
		{
			sum += this->Array_CPU_[i];
		}

		for (unsigned int i = 0; i < this->Row_ * this->Col_; i++)
		{
			this->Array_CPU_[i] /= sum;
		}
	}

	void Matrix::SVMPoolMatrix(const Matrix& source)
	{
		float * tmp_arr = source.GetMatrix();
		unsigned int row = source.GetRowSize(), col = source.GetColSize();
		if (false)//serial cpu better.
		{
			static const int TS = 2;
			concurrency::extent<2> e((int)row, (int)col);
			concurrency::array_view<const float, 2> other(e, source.GetMatrix());
			concurrency::array_view<float, 2> target((int)this->Row_, (int)this->Col_, this->Array_CPU_);
			target.discard_data();

			parallel_for_each(target.extent.tile<TS, TS>(), [=](concurrency::tiled_index<TS, TS> t_idx) restrict(amp)//target,other,col
			{

				tile_static float nums[TS][TS];
				nums[t_idx.local[0]][t_idx.local[1]] = other[t_idx.global];
				t_idx.barrier.wait_with_tile_static_memory_fence();
				float num = -FLT_MAX;
				for (int i = 0; i < 2; i++)
				{
					for (int j = 0; j < 2; j++)
					{
						if (num < nums[i][j])
						{
							num = nums[i][j];
						}
					}
				}
				target[t_idx.tile] = num;
			}
			);
			target.synchronize();
		}
		else
		{
			for (unsigned int r = 0; r < row; r += 2)
			{
				for (unsigned int c = 0; c < col; c += 2)
				{
					float num = -FLT_MAX;
					for (unsigned int x = 0; x < 2; x++)
					{
						for (unsigned int y = 0; y < 2; y++)
						{
							if (num < tmp_arr[(r + x) * col + c + y])
							{
								num = tmp_arr[(r + x) * col + c + y];
							}
						}
					}
					this->Array_CPU_[(r / 2) * (col / 2) + c / 2] = num;
				}
			}
		}
	}

	void Matrix::AVGPoolMatrix(const Matrix& source)
	{
		float * tmp_arr = source.GetMatrix();
		unsigned int row = source.GetRowSize(), col = source.GetColSize();
		for (unsigned int r = 0; r < row; r += 2)
		{
			for (unsigned int c = 0; c < col; c += 2)
			{
				float num = 0.0f;
				for (unsigned int x = 0; x < 2; x++)
				{
					for (unsigned int y = 0; y < 2; y++)
					{
						num += tmp_arr[(r + x) * col + c + y];
					}
				}
				this->Array_CPU_[(r / 2) * (col / 2) + c / 2] = num;
			}
		}
	}

	void Matrix::ActivateMatrix(const Matrix& source, unsigned int AFidx, float bias)
	{
		float* tmp_arr = source.GetMatrix();
		float* tmp_arr2 = this->Array_CPU_;

		switch (AFidx)
		{
		case 1://sigmoid
		{
			if (source.Optimizer(true, source.GetIdentity(), true) == GPU_)
			{
				concurrency::array_view<const float, 1> other((int)(this->Row_ * this->Col_), source.GetMatrix());
				concurrency::array_view<float, 1> target((int)(this->Row_ * this->Col_), this->Array_CPU_);
				parallel_for_each(target.extent, [=](concurrency::index<1> idx) restrict(amp)
				{
					target[idx] = 1.0f / (1.0f + concurrency::precise_math::expf(-other[idx] - bias));
				}
				);
				target.synchronize();
			}
			else
			{
				size_t size_n = size_t(this->Row_), size_w = size_t(this->Col_);
				concurrency::parallel_for(size_t(0), size_n, [&](size_t i)
				{
					for (size_t j = 0; j < size_w; j++)
					{
						tmp_arr2[i * size_w + j] = 1.0f / (1.0f + std::expf(-tmp_arr[i * size_w + j] - bias));
					}
				});
			}
			break;
		}

		case 2://tanh
		{
			if (source.Optimizer(true, source.GetIdentity(), true) == GPU_)
			{
				concurrency::array_view<const float, 1> other((int)(this->Row_ * this->Col_), source.GetMatrix());
				concurrency::array_view<float, 1> target((int)(this->Row_ * this->Col_), this->Array_CPU_);
				parallel_for_each(target.extent, [=](concurrency::index<1> idx) restrict(amp)
				{
					target[idx] = concurrency::precise_math::tanhf(other[idx] + bias);
				}
				);
				target.synchronize();
			}
			else
			{
				size_t size_n = size_t(this->Row_), size_w = size_t(this->Col_);
				concurrency::parallel_for(size_t(0), size_n, [&](size_t i)
				{
					for (size_t j = 0; j < size_w; j++)
					{
						tmp_arr2[i * size_w + j] = std::tanhf(tmp_arr[i * size_w + j] + bias);
					}
				});
			}
			break;
		}

		case 3://ReLU
		{
			if (source.Optimizer(true, source.GetIdentity(), true) == GPU_)
			{
				concurrency::array_view<const float, 1> other((int)(this->Row_ * this->Col_), source.GetMatrix());
				concurrency::array_view<float, 1> target((int)(this->Row_ * this->Col_), this->Array_CPU_);
				parallel_for_each(target.extent, [=](concurrency::index<1> idx) restrict(amp)
				{
					target[idx] = (other[idx] + bias >= 0.0f) ? (other[idx] + bias) : 0.0f;
				}
				);
				target.synchronize();
			}
			else
			{
				size_t size_n = size_t(this->Row_), size_w = size_t(this->Col_);
				concurrency::parallel_for(size_t(0), size_n, [&](size_t i)
				{
					for (size_t j = 0; j < size_w; j++)
					{
						tmp_arr2[i * size_w + j] = (tmp_arr[i * size_w + j] + bias > 0) ? (tmp_arr[i * size_w + j] + bias) : 0.0f;
					}
				});
			}
			break;
		}

		default://softmax
		{
			this->operator=(source);
			this->SoftMaxMatrix();
			break;
		}
		}
	}

	void Matrix::DerivativeMatrix(const Matrix& source, unsigned int AFidx, float bias)
	{
		float* tmp_arr = source.GetMatrix();
		float* tmp_arr2 = this->Array_CPU_;

		switch (AFidx)
		{
		case 1:
		{
			if (source.Optimizer(true, source.GetIdentity(), true) == GPU_)
			{
				concurrency::array_view<const float, 1> other((int)(this->Row_ * this->Col_), source.GetMatrix());
				concurrency::array_view<float, 1> target((int)(this->Row_ * this->Col_), this->Array_CPU_);
				parallel_for_each(target.extent, [=](concurrency::index<1> idx) restrict(amp)
				{
					target[idx] = concurrency::precise_math::expf(-other[idx] - bias) / concurrency::precise_math::powf(1.0f + concurrency::precise_math::expf(-other[idx] - bias), 2);
				}
				);
				target.synchronize();
			}
			else
			{
				size_t size_n = size_t(this->Row_), size_w = size_t(this->Col_);
				concurrency::parallel_for(size_t(0), size_n, [&](size_t i)
				{
					for (size_t j = 0; j < size_w; j++)
					{
						tmp_arr2[i * size_w + j] = std::expf(-tmp_arr[i * size_w + j] - bias) / std::powf(1.0f + std::expf(-tmp_arr[i * size_w + j] - bias), 2);
					}
				});
			}
			break;
		}

		case 2:
		{
			if (source.Optimizer(true, source.GetIdentity(), true) == GPU_)
			{
				concurrency::array_view<const float, 1> other((int)(this->Row_ * this->Col_), source.GetMatrix());
				concurrency::array_view<float, 1> target((int)(this->Row_ * this->Col_), this->Array_CPU_);
				parallel_for_each(target.extent, [=](concurrency::index<1> idx) restrict(amp)
				{
					target[idx] = 1.0f - concurrency::precise_math::powf(concurrency::precise_math::tanhf(other[idx] + bias), 2);
				}
				);
				target.synchronize();
			}
			else
			{
				size_t size_n = size_t(this->Row_), size_w = size_t(this->Col_);
				concurrency::parallel_for(size_t(0), size_n, [&](size_t i)
				{
					for (size_t j = 0; j < size_w; j++)
					{
						tmp_arr2[i * size_w + j] = std::powf(std::tanhf(tmp_arr[i * size_w + j] + bias), 2);
					}
				});
			}
			break;
		}

		default:
		{
			if (source.Optimizer(true, source.GetIdentity(), true) == GPU_)
			{
				concurrency::array_view<const float, 1> other((int)(this->Row_ * this->Col_), source.GetMatrix());
				concurrency::array_view<float, 1> target((int)(this->Row_ * this->Col_), this->Array_CPU_);
				parallel_for_each(target.extent, [=](concurrency::index<1> idx) restrict(amp)
				{
					target[idx] = (other[idx] + bias >= 0.0f) ? 1.0f : 0.0f;
				}
				);
				target.synchronize();
			}
			else
			{
				size_t size_n = size_t(this->Row_), size_w = size_t(this->Col_);
				concurrency::parallel_for(size_t(0), size_n, [&](size_t i)
				{
					for (size_t j = 0; j < size_w; j++)
					{
						tmp_arr2[i * size_w + j] = (tmp_arr[i * size_w + j] + bias >= 0.0f) ? 1.0f : 0.0f;
					}
				});
			}
			break;
		}
		}
	}

	void Matrix::DotMatrices(const Matrix& rhs)
	{
		float* tmp = rhs.GetMatrix();
		for (unsigned int r = 0; r < this->Row_; r++)
			for (unsigned int c = 0; c < this->Col_; c++)
				this->Array_CPU_[r * this->Col_ + c] *= tmp[r * this->Col_ + c];

	}

	void Matrix::ReduceMatricesToOne(shared_vector& rhs)
	{
		int R_size = (int)rhs.at(0)->GetRowSize();
		int C_size = (int)rhs.at(0)->GetColSize();
		for (int s = 0; s < (int)rhs.GetSize(); s++)
		{
			float* tmp = rhs.at(s)->GetMatrix();
			for (int r = 0; r < R_size; r++)
				for (int c = 0; c < C_size; c++)
					this->Array_CPU_[s * R_size * C_size + r * C_size + c] = tmp[r * C_size + c];
		}
	}

	void Matrix::Norm()
	{
		float sum = 0.0f;

		for (unsigned int i = 0; i < this->Row_ * this->Col_; i++)
		{
			sum += this->Array_CPU_[i] * this->Array_CPU_[i];
		}

		sum = sqrt(sum);
		if (sum != 0.0f)
		{
			for (unsigned int i = 0; i < this->Row_ * this->Col_; i++)
			{
				this->Array_CPU_[i] /= sqrt(sum);
			}
		}
	}

	float Matrix::FindMaxVal() const
	{
		float tmp = 0.00001f;
		for (unsigned int r = 0; r < this->Row_; r++)
		{
			for (unsigned int c = 0; c < this->Col_; c++)
			{
				if (this->Array_CPU_[r * this->Col_ + c] > tmp)
				{
					tmp = this->Array_CPU_[r * this->Col_ + c];
				}
			}
		}

		return tmp;
	}

	Matrix Matrix::ColumnSum() const
	{
		Matrix tmp(this->Row_, 1, Zeroes);
		float sum = 0.0;
		for (unsigned int r = 0; r<this->Row_; r++)
		{
			for (unsigned int c = 0; c < this->Col_; c++)
			{
				sum += this->Array_CPU_[r * this->Col_ + c];
			}
			tmp.operator()(r, 0) = sum;
			sum = 0;
		}

		return tmp;
	}

	Matrix Matrix::RowSum(unsigned int SheetSize) const
	{
		Matrix tmp(SheetSize, 1, Zeroes);
		unsigned int div_const = this->Row_ / SheetSize;// if the matrix size is 400x1 and the SheetSize is 25, then div_const = 16.
		for (unsigned int r = 0; r < div_const; r++)
		{
			for (unsigned int s = 0; s < SheetSize; s++)
			{
				tmp(s, 0) += this->Array_CPU_[r * SheetSize + s];
			}
		}

		return tmp;
	}

	void Matrix::MatrixRotation_180()
	{
		float* tmp_arr = new float[this->Row_ * this->Col_];
		for (int r = (int)(this->Row_) - 1; r >= 0; r--)
		{
			for (int c = (int)(this->Col_) - 1; c >= 0; c--)
			{
				tmp_arr[(this->Row_ - 1 - r) * this->Col_ + (this->Col_ - 1 - c)] = this->Array_CPU_[r * this->Col_ + c];
			}
		}
		delete[] this->Array_CPU_;
		this->Array_CPU_ = tmp_arr;
	}

	float Matrix::MatrixSumAllElements() const
	{
		float sum = 0.0f;
		for (unsigned int r = 0; r < this->Row_; r++)
		{
			for (unsigned int c = 0; c < this->Col_; c++)
			{
				sum += this->Array_CPU_[r * this->Col_ + c];
			}
		}

		return sum;
	}

	/*Matrix Matrix::Inverse()
	{
	Checker(4);
	Matrix tmp(this->Row_,this->Col_,0,Zeroes);
	double** tmp2 = this->GaussianElimination();
	for (size_t r = 0; r < this->Row_; r++)
	for (size_t c = 0; c < this->Col_; c++)
	tmp(r,c) = (float)tmp2[r][c];

	return tmp;
	}*/

	/*float Matrix::Determinant() const
	{
	Checker(3);
	double** tmp = this->GaussianElimination(true);
	double sum = 0.0;
	for (size_t r = 0; r < this->Row_; r++)
	sum *= tmp[r][r];

	delete tmp;
	return (float)sum;
	}*/

	/*Matrix Matrix::Eigenvalues() const
	{
	Checker(4);
	double** tmp2 = this->GaussianElimination(true);
	Matrix tmp(1, this->Col_, 0, Zeroes);

	for (size_t c = 0; c < this->Col_; c++)
	tmp(1,c) = tmp2[c][c];

	delete tmp2;
	return tmp;
	}*/

	/*Matrix* Matrix::GaussianElimination(bool isInverse)
	{
		float EPS = 0.000001f;
		Matrix* IdentityMat = new Matrix(this->Row_, this->Col_, DontCare);
		INC_ROW_LOOP(0,this->Row_)
		{
			INC_COL_LOOP(0,this->Col_)
			{
				(*IdentityMat)(r,c) = 0.0f;
			}
			(*IdentityMat)(r, r) = 1.0f;
		}

		for (int i = 0; i < this->Row_; i++)
		{
			if (this->Array_CPU_[i * this->Col_ + i] == 0)
			{
				bool pivot_found = false;
				for (int j = i + 1; j < this->Row_; j++)
				{
					if (this->Array_CPU_[j * this->Col_ + i] != 0)
					{
						this->SwapRows(i,j);
						pivot_found = true;
						break;
					}
				}

				if (!pivot_found)
				{
					IdentityMat->~Matrix();
					throw exceptionh::ExceptionHandler("The matrix isn't inversable.");
				}
			}

			float Scale = this->Array_CPU_[i * this->Col_ + i];
			concurrency::array_view<float,2> A(1,this->Col_,this->Array_CPU_ + i * this->Col_);
			concurrency::parallel_for_each(A.extent.tile<1,2>(), [=](concurrency::tiled_index<1,2> t_idx) restrict(amp)
			{
				A[t_idx] /= Scale;
			});
			A.synchronize();

			for (unsigned int j = 0; j < this->Row_; j++)
			{
				if (j == i)
				{
					continue;
				}

				float num = A[j][i];
				concurrency::array_view<float,2> A(this->Col_, this->Array_CPU_ + i * this->Col_);
				concurrency::array_view<float,2> B(this->Col_, this->Array_CPU_ + j * this->Col_);
				concurrency::parallel_for_each(A.extent.tile<1,2>(), [=](concurrency::tiled_index<1,2> t_idx) restrict(amp)
				{
					B[t_idx] -= num * A(0,t_idx.global[1]);
				});
				A.synchronize();
			}
		}

		if (isInverse)
		{
			IdentityMat->~Matrix();
			return nullptr;
		}

		return IdentityMat;
	}*/

	void Matrix::SwapRows(int row1, int row2)
	{
		concurrency::array_view<float,2> target1(1,this->Col_,this->Array_CPU_ + row1 * this->Col_);
		concurrency::array_view<float,2> target2(1,this->Col_,this->Array_CPU_ + row2 * this->Col_);
		concurrency::parallel_for_each(target1.extent.tile<1,2>(), [=](concurrency::tiled_index<1,2> t_idx) restrict(amp)
		{
			float tmp = target1[t_idx];
			target1[t_idx] = target2[t_idx];
			target2[t_idx] = tmp;
		});
		target1.synchronize();
		target2.synchronize();
	}

	void Matrix::SaveMatrix(const std::string& _file) const
	{
		std::ofstream myfile;
		myfile.open(_file.c_str(), std::ios::app);//"MatrixData.txt" = _file
		time_t end_time = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
		myfile << "Created date:" << time(&end_time) << std::endl;
		for (unsigned int r = 0; r < this->Row_; r++)
			for (unsigned int c = 0; c < this->Col_; c++)
				myfile << this->Array_CPU_[r * this->Col_ + c] << std::endl;

		myfile.close();
	}

	void Matrix::LoadMatrix(const std::string& _file)
	{
		std::ifstream myfile(_file);//"MatrixData.txt" = _file
		std::string Matrixdata;
		std::getline(myfile, Matrixdata);
		std::getline(myfile, Matrixdata);
		std::string().swap(Matrixdata);

		for (unsigned int r = 0; r < this->Row_; r++)
			for (unsigned int c = 0; c < this->Col_; c++)
			{
				std::getline(myfile, Matrixdata);
				this->Array_CPU_[r * this->Col_ + c] = (float)atof(Matrixdata.c_str());
				std::string().swap(Matrixdata);
			}
	}

	void Matrix::PushBack(float Val)
	{
		unsigned int size = this->Row_ * this->Col_ + 1;
		float* Tmp = new float[size];

		for (unsigned int i = 0; i < size; i++)
		{
			if (i < size - 1)
				Tmp[i] = this->Array_CPU_[i];
			else
				Tmp[i] = Val;
		}

		delete this->Array_CPU_;
		this->Array_CPU_ = new float[size];

		for (unsigned int i = 0; i < size; i++)
		{
			this->Array_CPU_[i] = Tmp[i];
		}

		delete Tmp;
	}

	void Matrix::PopBack()
	{
		unsigned int size = this->Row_ * this->Col_ - 1;
		float* Tmp = new float[size];

		for (unsigned int i = 0; i < size; i++)
		{
			Tmp[i] = this->Array_CPU_[i];
		}

		delete this->Array_CPU_;
		this->Array_CPU_ = new float[size];

		for (unsigned int i = 0; i < size; i++)
		{
			this->Array_CPU_[i] = Tmp[i];
		}

		delete Tmp;
	}

	Device_ Matrix::Optimizer(bool SelfOperations, Identity_ idt, bool Selfarithmetic, bool ConvOp) const
	{
		if (ConvOp == true)
		{
			if (this->Row_ * this->Col_ <= FirstLimit)
			{
				return CPU_parallel_;
			}
			else
			{
				return GPU_;
			}
		}

		if (SelfOperations == true)
		{
			if (Selfarithmetic == false)
			{
				if (this->Row_ * this->Col_ <= FourthLimit)
				{
					return CPU_;
				}
				else
				{
					return GPU_;
				}
			}
			else
			{
				if (this->Row_ * this->Col_ <= SecondLimit)
				{
					return CPU_parallel_;
				}
				else
				{
					return GPU_;
				}
			}
		}

		else
		{
			if (idt == IsAMatrix && this->Idt_ == IsAMatrix)// both entities are matrices.
			{
				if (this->Row_ * this->Col_ * 128 <= ThirdLimit)
				{
					return CPU_;
				}
				else
				{
					return GPU_;
				}
			}

			else // one of the entities is a vector.
			{
				if (this->Row_ * this->Col_ <= SecondLimit)
				{
					return CPU_;
				}
				else
				{
					return CPU_parallel_;
				}
			}
		}
	}

	Matrix Log(const Matrix& rhs)
	{
		Matrix tmp(rhs);
		unsigned int row = tmp.GetRowSize(), col = tmp.GetColSize();
		float* tmp_arr = tmp.GetMatrix();
		float* tmp_arr2 = rhs.GetMatrix();

		if (tmp.Optimizer(true, tmp.GetIdentity(), true) == GPU_)
		{
			concurrency::extent<1> e((int)(row * col));
			concurrency::array_view<const float, 1> other(e, tmp_arr2);
			concurrency::array_view<float, 1> target(e, tmp_arr);
			concurrency::parallel_for_each(target.extent, [=](concurrency::index<1> idx) restrict(amp)
			{
				target[idx] = concurrency::precise_math::log10f(other[idx]);
			});
			target.synchronize();
		}
		else
		{
			size_t size_n = size_t(row), size_w = size_t(col);
			concurrency::parallel_for(size_t(0), size_n, [&](size_t i)
			{
				for (size_t j = 0; j < size_w; j++)
				{
					tmp_arr[i * size_w + j] = std::log10f(tmp_arr2[i * size_w + j]);
				}
			});
		}
		return tmp;
	}

	Matrix ReLU(const Matrix& rhs)
	{
		Matrix tmp(rhs);
		unsigned int row = tmp.GetRowSize(), col = tmp.GetColSize();
		float* tmp_arr = tmp.GetMatrix();
		float* tmp_arr2 = rhs.GetMatrix();

		if (tmp.Optimizer(true, tmp.GetIdentity(), true) == GPU_)
		{
			concurrency::extent<1> e((int)(row * col));
			concurrency::array_view<const float, 1> other(e, tmp_arr2);
			concurrency::array_view<float, 1> target(e, tmp_arr);
			concurrency::parallel_for_each(target.extent, [=](concurrency::index<1> idx) restrict(amp)
			{
				target[idx] = (other[idx] > 0) ? other[idx] : 0.0f;
			});
			target.synchronize();
		}
		else
		{
			size_t size_n = size_t(row), size_w = size_t(col);
			concurrency::parallel_for(size_t(0), size_n, [&](size_t i)
			{
				for (size_t j = 0; j < size_w; j++)
				{
					tmp_arr[i * size_w + j] = (tmp_arr2[i * size_w + j] > 0) ? tmp_arr2[i * size_w + j] : 0.0f;
				}
			});
		}
		return tmp;
	}

	Matrix Tanh(const Matrix& rhs)
	{
		Matrix tmp(rhs);
		unsigned int row = tmp.GetRowSize(), col = tmp.GetColSize();
		float* tmp_arr = tmp.GetMatrix();
		float* tmp_arr2 = rhs.GetMatrix();

		if (tmp.Optimizer(true, tmp.GetIdentity(), true) == GPU_)
		{
			concurrency::extent<1> e((int)(row * col));
			concurrency::array_view<const float, 1> other(e, tmp_arr2);
			concurrency::array_view<float, 1> target(e, tmp_arr);
			concurrency::parallel_for_each(target.extent, [=](concurrency::index<1> idx) restrict(amp)
			{
				target[idx] = concurrency::precise_math::tanhf(other[idx]);
			});
			target.synchronize();
		}
		else
		{
			size_t size_n = size_t(row), size_w = size_t(col);
			concurrency::parallel_for(size_t(0), size_n, [&](size_t i)
			{
				for (size_t j = 0; j < size_w; j++)
				{
					tmp_arr[i * size_w + j] = std::tanhf(tmp_arr2[i * size_w + j]);
				}
			});
		}
		return tmp;
	}

	Matrix SoftMaxMatrix(const Matrix& rhs)
	{
		Matrix tmp(rhs);
		unsigned int s = rhs.GetColSize() * rhs.GetRowSize();
		concurrency::array_view<float, 1> target((int)s, tmp.GetMatrix());
		float sum = 0.0f;
		parallel_for_each(target.extent, [=](concurrency::index<1> idx) restrict(amp)
		{
			target[idx] = concurrency::precise_math::expf(target[idx]);
		});
		target.synchronize();

		float* tmp_mat = tmp.GetMatrix();
		for (unsigned int i = 0; i < s; i++)
		{
			sum += tmp_mat[i];
		}

		for (unsigned int i = 0; i < s; i++)
		{
			tmp_mat[i] /= sum;
		}

		return tmp;
	}

	Matrix AvgMatrix(const Matrix& rhs)
	{
		Matrix tmp(rhs);
		float sum = 0.0f;
		float* tmp_array = rhs.GetMatrix();
		unsigned int R_s = rhs.GetRowSize(), C_s = rhs.GetColSize();
		for (unsigned int r = 0; r < R_s; r++)
		{
			for (unsigned int c = 0; c < C_s; c++)
			{
				sum += tmp_array[r * C_s + c];
			}
		}

		for (unsigned int r = 0; r < rhs.GetRowSize(); r++)
		{
			for (unsigned int c = 0; c < rhs.GetColSize(); c++)
			{
				tmp(r, c) /= sum;
			}
		}

		return tmp;
	}

	int ReverseInt(int i)
	{
		unsigned char ch1, ch2, ch3, ch4;
		ch1 = i & 255;
		ch2 = (i >> 8) & 255;
		ch3 = (i >> 16) & 255;
		ch4 = (i >> 24) & 255;
		return ((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
	}

	void read_Mnist(const std::string& filename, shared_vector& vec, int Number_of_imgs)
	{
		std::ifstream file(filename, std::ios::binary);
		if (file.is_open())
		{
			int magic_number = 0;
			int number_of_images = 0;
			int n_rows = 0;
			int n_cols = 0;
			file.read((char*)&magic_number, sizeof(magic_number));
			magic_number = ReverseInt(magic_number);
			file.read((char*)&number_of_images, sizeof(number_of_images));
			number_of_images = ReverseInt(number_of_images);
			file.read((char*)&n_rows, sizeof(n_rows));
			n_rows = ReverseInt(n_rows);
			file.read((char*)&n_cols, sizeof(n_cols));
			n_cols = ReverseInt(n_cols);
			int stop_lim = (Number_of_imgs < number_of_images) ? Number_of_imgs : number_of_images;

			for (int i = 0; i < stop_lim; i++)
			{
				for (int r = 0; r < n_rows; r++)
				{
					for (int c = 0; c < n_cols; c++)
					{
						unsigned char tmp = 0;
						file.read((char*)&tmp, sizeof(tmp));
						(*vec.at(i))(r + 2, c + 2) = ((255.0f - (float)tmp) / 200.0f) - 0.1f;
					}
				}
			}
		}
	}

	void read_Mnist_Label(const std::string filename, std::vector<float>& vec, int Number_of_imgs)
	{
		std::ifstream file(filename, std::ios::binary);
		if (file.is_open())
		{
			int magic_number = 0;
			int number_of_images = 0;
			file.read((char*)&magic_number, sizeof(magic_number));
			magic_number = ReverseInt(magic_number);
			file.read((char*)&number_of_images, sizeof(number_of_images));
			number_of_images = ReverseInt(number_of_images);
			int stop_lim = (Number_of_imgs < number_of_images) ? Number_of_imgs : number_of_images;

			for (int i = 0; i < stop_lim; i++)
			{
				unsigned char tmp = 0;
				file.read((char*)&tmp, sizeof(tmp));
				vec.at(i) = (float)tmp;
			}
		}
	}

}
