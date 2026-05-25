#ifndef OPTIMIZER_HPP
#define OPTIMIZER_HPP

#include <boost/serialization/nvp.hpp>
#include <boost/serialization/string.hpp>

#include <ostream>
#include <string>

#include "exccnn/dllexport.h"
#include "typedef.hpp"
#include "functions.hpp"

namespace cnn {

/* Provide function pointers to activation function and derivative of activation function */
class EXCCNN_PUBLIC optimizer {
public:
	typedef real_type (*FunctionPointer)(real_type);

	std::string name;
	// FIXME: activate and derivative_funcptr are lost on serialization
	FunctionPointer activate;
	FunctionPointer derivative_funcptr;

	// TODO: split into save/load so we can map activate + derivative_funcptr
	// to the known combinations of activation functions + their derivatives
	template <typename Ar>
	void serialize(Ar& ar, unsigned /*version*/) {
		ar & BOOST_SERIALIZATION_NVP(name);
	}

	/**
	 * Default ctor.
	 *
	 * This constructs with the `"linear"`activation name and function pointers.
	 */
	optimizer();

	/**
	 * Ctor.
	 *
	 * Initializes with user-specified choice of name and function pointers.
	 *
	 * @todo This does correctly constrain the inputs to the known combinations.
	 *
	 * @param name Activation name
	 * @param funcptr Activation function pointer
	 * @param derivative_funcptr Activation derivative function pointer
	 */
	optimizer(
		std::string name,
		FunctionPointer funcptr,
		FunctionPointer derivative_funcptr);
};

/**
 * Write the optimizer to the output stream.
 *
 * This streams `"optimizer = <name>"`, where `<name>` is the object name.
 *
 * @param out Output stream
 * @param op Optimizer object
 */
EXCCNN_PUBLIC
std::ostream& operator<<(std::ostream& out, const optimizer& op);

}  // namespace cnn

#endif  // OPTIMIZER_HPP
