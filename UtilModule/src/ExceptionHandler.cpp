#include "ExceptionHandler.h"

namespace exceptionh {

	ExceptionHandler::ExceptionHandler(const char* message)
		:
		msg_(message)
	{}

	ExceptionHandler::ExceptionHandler(const std::string& message)
		:
		msg_(message)
	{}

	ExceptionHandler::~ExceptionHandler() throw ()
	{}

	const char* ExceptionHandler::what() const throw ()
	{
		return msg_.c_str();
	}
}