# Huawei CodeCraft 2022 Solutions

*2022*华为软件精英挑战赛

## Requirements

1. Ubuntu 20.04
2. python3.8 and please refer the `requiremetns.txt`

## Solutions

1. FFD

   ```bash
   python run.py --strategy simple
   ```

2. Linear Programming 

   ```bash
   cd src && python simpleSolution.py --linear normal
   ```

3. Linear Programming (more tight)

   ```bash
   cd src && python simpleSolution.py --linear improved
   ```

4. Astar

   ```bash
   python run.py --strategy astar
   ```

The details can be found in the report.

## Benchmark 

Dataset: data1/ (Provided by Huawei official)

- Simple: 177,272
- linear Programing: 7,958 in 300 seconds, at least 5,641 without time limited
- Astar: 27,068

Dataset: data2/ (A big size problem)

- Simple: 627,877
- linear Programing: 615,130
- Astar: 617,287

## Co-Authors

@[Raibows](https://github.com/Raibows) @[tanyingkongxin](https://github.com/tanyingkongxin ) @Jian Wang