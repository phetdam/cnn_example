#ifndef CONNECTION_HPP
#define CONNECTION_HPP

#include <boost/serialization/nvp.hpp>
#include <boost/serialization/vector.hpp>

#include <iosfwd>
#include <vector>

#include "exccnn/dllexport.h"
#include "typedef.hpp"
#include "connector.hpp"

namespace cnn {

class EXCCNN_PUBLIC Connection {
public:
  /**
   * Boost serialization function for name-value serialization.
   *
   * @tparam Ar Boost.Serialization input/output archive type
   *
   * @param ar Boost.Serialiation input/output archive
   */
	template <typename Ar>
	void serialize(Ar& ar, unsigned /*version*/)
  {
		ar & BOOST_SERIALIZATION_NVP(weights);
	}

  /**
   * Default ctor.
   */
  Connection() = default;

  /**
   * Ctor.
   *
   * @param _weights `Connector` weights to initialize with
   */
  Connection(std::vector<Connector> _weights);

  // forward update (no-op)
  void update_forward(real_type /*input*/);

  // backward update (no-op)
  void update_backward(
    real_type /*previous*/,
    real_type /*expected*/,
    real_type /*next*/);

  /**
   * Compare two `Connection` objects for equality.
   *
   * Currently tbis implementation only checks that the weights are the same
   * size and the values of each `Connector` in the weights vector.
   *
   * @param con `Connection` to compare against
   */
  bool operator==(const Connection& con) const noexcept;

  // connection weights
  std::vector<Connector> weights;
};

/**
 * Write the `Connection` to the output stream.
 *
 * This simply streams all the `Connector` items in the weights.
 *
 * @param out Output stream
 * @param con Network connection
 */
EXCCNN_PUBLIC
std::ostream& operator<<(std::ostream& out, const Connection& con);

}  // namespace cnn

#endif  // CONNECTION_HPP
