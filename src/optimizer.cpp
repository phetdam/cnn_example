#include "optimizer.hpp"

#include <ostream>
#include <utility>

#include "functions.hpp"

namespace cnn {

optimizer::optimizer()
  : name{"linear"}, activate{linear}, derivative_funcptr{derivative_linear}
{}

optimizer::optimizer(
  std::string name,
  FunctionPointer funcptr,
  FunctionPointer derivative_funcptr)
  : name{std::move(name)},
    activate{funcptr},
    derivative_funcptr{derivative_funcptr}
{}

std::ostream&
operator<<(std::ostream& out, const optimizer& op)
{
  return out << "optimizer = " << op.name;
}

}  // namespace cnn
