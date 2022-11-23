# TPVFormer: An academic alternative to Tesla's Occupancy Network
Under construction.

Paper, code, and models are coming soon!

## Demo

![demo](./assets/demo.gif)

![legend](./assets/legend.png)

### A full demo video can be downloaded [here](https://cloud.tsinghua.edu.cn/f/594cadf14ae949228df1/).

## Comparisons with Tesla's Occupancy Network

|                          | **Tesla's Occupancy Network**        | **Our TPVFormer**                |
| ------------------------ | ------------------------------------ | -------------------------------- |
| **Volumetric Occupancy** | Yes                                  | Yes                              |
| **Occupancy Semantics**  | Yes                                  | Yes                              |
| **#Semantics**           | >= 5                                 | **16**                           |
| **Input**                | 8 camera images                      | 6 camera images                  |
| **Training Supervision** | Dense 3D reconstruction              | **Sparse LiDAR semantic labels** |
| **Training Data**        | ~1,440,000,000 frames                | **28,130 frames**                |
| **Arbitrary Resolution** | Yes                                  | Yes                              |
| **Video Context**        | **Yes**                              | Not yet                          |
| **Training Time**        | ~100,000 gpu hours                   | **~300 gpu hours**               |
| **Inference Time**       | **~10 ms on the Tesla FSD computer** | ~290 ms on a single A100         |



## Visualizations

![](./assets/vis1.png)

![](./assets/vis2.png)

![](./assets/vis3.png)

![](./assets/vis4.png)

## Lidar Segmentation Results

![](./assets/results.png)
