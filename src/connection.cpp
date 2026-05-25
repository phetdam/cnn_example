#include "connection.hpp"

#include <ostream>
#include <utility>

#include "typedef.hpp"

namespace cnn {

Connection::Connection(std::vector<Connector> _weights)
  : weights{std::move(_weights)}
{}

void
Connection::update_forward(real_type /* input */)
{
  // no-op
}

void
Connection::update_backward(
  real_type /*previous*/,
  real_type /*expected*/,
  real_type /*next*/)
{
  // no-op
}

bool
Connection::operator==(const Connection& con) const noexcept
{
  return weights.size() == con.weights.size();
}

std::ostream& operator<<(std::ostream& out, const Connection& con)
{
  for (const auto& weight : con.weights)
    out << weight;
  return out;
}

}  // namespace cnn
