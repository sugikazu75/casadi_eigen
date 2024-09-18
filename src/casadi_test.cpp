#include <casadi/casadi.hpp>

int main()
{
  // variable
  casadi::SX x = casadi::SX::sym("x", 3);

  bool verbose = true;

  std::cout << x << std::endl;
  std::cout << std::endl;

  casadi::SX cost = dot(x, x);
  std::cout << cost << std::endl;
  std::cout << std::endl;

  casadi::SX grad = gradient(cost, x);
  std::cout << grad << std::endl;
  std::cout << std::endl;

  casadi::SX x0 = casadi::SX(3,1);
  x0(0) = 0.5;
  x0(1) = x(1);
  x0(2) = x(2);
  casadi::Function f = casadi::Function("f", {x}, {cost});
  casadi::SX ret = f(x0)[0]; // necessary to get first element.
  std::cout << ret << std::endl;
  std::cout << std::endl;

  casadi::SX G = casadi::SX(2, 1);
  G(0) = 1.0;
  G(1) = 1.0;
  std::cout << G << std::endl;
  std::cout << std::endl;

  casadi::SX x_opt = casadi::SX(2, 1);
  x_opt(0) = x(1);
  x_opt(1) = x(2);
  std::cout << x_opt << std::endl;

  casadi::SX constraints = dot(G, x_opt);
  std::cout << constraints << std::endl;
  std::cout << std::endl;

  casadi::SXDict nlp = {{"x", casadi::SX::vertcat({x_opt})}, {"f", ret}, {"g", constraints} };

  casadi::Function S = casadi::nlpsol("S", "ipopt", nlp);
  casadi::DM initial_x = casadi::DM({1.234, 5.678});
  auto res = S(casadi::DMDict{{"x0", initial_x}, {"lbg", casadi::SX(1.0)}, {"ubg", casadi::SX(1.0)}});
  std::cout << res << std::endl;

}
