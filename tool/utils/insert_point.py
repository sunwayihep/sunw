# lambda = 3 / const
def from_lambda_to_const(lambda_) :
  return 3 / lambda_


def interpolate(x: list, y: list, threshold_x: float, threshold_y: float) ->list :
  res = []
  for i in range(len(x)) :
    if i == 0:
      continue
    dx = x[i] - x[i - 1]
    dy = y[i] - y[i - 1]
    # print(f'i = {i}, dy / dx = {abs(dy / dx)}, {abs(dx)}, {threshold_x}, {threshold_y}')
    # print(f'dx = {dx}, lambda[{i}] = {x[i]}, lambda_[{i-1}] = {x[i - 1]}');

    if abs(dy / dx) > threshold_y and abs(dy) > 0.03 and abs(dx) > threshold_x:
      # print(f'############### i = {i}, dy / dx = {abs(dy / dx)}')
      res.append((from_lambda_to_const(x[i]) + from_lambda_to_const(x[i-1]))/2) # 插点
  print(res)
  return res