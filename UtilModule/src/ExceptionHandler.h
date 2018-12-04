#ifndef _EXCEPTION_HANDLER_H_
#define _EXCEPTION_HANDLER_H_

#include "IncluderLib.h"

namespace exceptionh {
	class ExceptionHandler : public std::exception
	{
	public:
		/* Constructor (C strings).
		*		@param message C-style string error message.
		*       The string contents are copied upon construction.
		*       Hence, responsibility for deleting the char* lies
		*       with the caller.
		*/
		explicit ExceptionHandler(const char* message);

		/* Constructor (C++ STL strings).
		*		@param message The error message.
		*/
		explicit ExceptionHandler(const std::string& message);

		/* Destructor.
		*		Virtual to allow for subclassing.
		*/

		virtual ~ExceptionHandler() throw ();

		/* Returns a pointer to the (constant) error description.
		*		@return A pointer to a const char*. The underlying memory
		*       is in posession of the Exception object. Callers must
		*       not attempt to free the memory.
		*/
		virtual const char* what() const throw ();

	protected:
		std::string msg_;
	};

	//chosing the amount of images that user want.
	class Excep_No1 : public std::exception
	{
	public:
		explicit Excep_No1(const char* message) { std::cout << message << std::endl; };
		explicit Excep_No1(const std::string& message) { std::cout << message << std::endl; };
	};

	//chosing custom data set.
	class Excep_No2 : public std::exception
	{
	public:
		explicit Excep_No2(const char* message) { std::cout << message << std::endl; };
		explicit Excep_No2(const std::string& message) { std::cout << message << std::endl; };
	};

	//loading 5000 images for testing.
	class Excep_No3 : public std::exception
	{
	public:
		explicit Excep_No3(const char* message) { std::cout << message << std::endl; };
		explicit Excep_No3(const std::string& message) { std::cout << message << std::endl; };
	};
}

#endif
