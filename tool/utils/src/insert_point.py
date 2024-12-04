def interpolate(x: list, y: list, threshold_x: float, threshold_slope: float) ->list :
  res = []
  for i in range(len(x)) :
    if i == 0:
      continue
    dx = x[i] - x[i - 1]
    dy = y[i] - y[i - 1]

    if abs(dy / dx) > threshold_slope and abs(dy) > 0.03 and abs(dx) > threshold_x:
      res.append((x[i] + x[i-1])/2) # 插点

  print(res)
  return res