from typing import Generator
import math

def divide(num, k_parts) -> list[int]:
  s = num
  sk = s // k_parts
  v = 0
  res: list[int] = []
  while v < s:
      res.append(v)
      v += sk
  return res

def determine_split(num: int) -> int:
  return int(math.ceil(math.pow(num, 1/3)))

def combine(divs: list[list]) -> Generator[str, None, None]:
  assert len(divs) > 0
  if len(divs) == 1:
    for obj in divs[0]:
      yield obj
  else:
    for obj in divs[0]:
      for rest in combine(divs[1:]):
        yield obj + rest

def main():
  num = 22
  k = determine_split(num)
  int_divs = [divide(256, k)] * 3
  hex_divs = [[hex(c)[2:].rjust(2, '0') for c in div] for div in int_divs]
  combs = list(combine(hex_divs))
  assert len(combs) >= num
  print(combs[:num])

if __name__:
  main()
