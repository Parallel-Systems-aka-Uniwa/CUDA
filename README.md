<p align="center">
  <img src="https://www.especial.gr/wp-content/uploads/2019/03/panepisthmio-dut-attikhs.png" alt="UNIWA" width="150"/>
</p>

<p align="center">
  <strong>UNIVERSITY OF WEST ATTICA</strong><br>
  SCHOOL OF ENGINEERING<br>
  DEPARTMENT OF COMPUTER ENGINEERING AND INFORMATICS
</p>

---

<p align="center">
  <strong>Parallel Systems</strong>
</p>

<h1 align="center">
  Parallel Computing using CUDA
</h1>

<p align="center">
  <strong>Vasileios Evangelos Athanasiou</strong><br>
  Student ID: 19390005
</p>

<p align="center">
  <a href="https://github.com/Ath21" target="_blank">GitHub</a> ·
  <a href="https://www.linkedin.com/in/vasilis-athanasiou-7036b53a4/" target="_blank">LinkedIn</a>
</p>

<hr/>

<p align="center">
  <strong>Supervision</strong>
</p>

<p align="center">
  Supervisor: Vasileios Mamalis, Professor
</p>
<p align="center">
  <a href="https://ice.uniwa.gr/en/emd_person/vassilios-mamalis/" target="_blank">UNIWA Profile</a>
</p>


<p align="center">
  Co-supervisor: Michalis Iordanakis, Special Technical Laboratory Staff
</p>

<p align="center">
  <a href="https://scholar.google.com/citations?user=LiVuwVEAAAAJ&hl=en" target="_blank">UNIWA Profile</a>
</p>

</hr>

<p align="center">
  Athens, February 2025
</p>


---

# Parallel Computing using CUDA

This repository implements **parallel operations on 2D integer arrays** using **CUDA** for high-performance GPU computation. The project was developed as part of the **Parallel Systems** course at the **University of West Attica**.

---

## Table of Contents

| Section | Folder/File | Description |
|------:|-------------|-------------|
| 1 | `assign/` | Assignment material for the CUDA workshop |
| 1.1 | `assign/_Par_Sys_Ask_2-1_2024-25.pdf` | Assignment description in English |
| 1.2 | `assign/_Παρ_Συσ_Ασκ_2-1_2024-25.pdf` | Assignment description in Greek |
| 2 | `docs/` | Documentation for parallel computing using CUDA |
| 2.1 | `docs/Parallel-Computig-using-CUDA.pdf` | English documentation for CUDA parallel computing |
| 2.2 | `docs/Παράλληλος-Υπολογισμός-με-CUDA.pdf` | Greek documentation for CUDA parallel computing |
| 3 | `src/` | Source code, input/output files, and CUDA implementation |
| 3.1 | `src/cuda1.cu` | Main CUDA program |
| 3.2 | `src/A/` | Input data files for CUDA exercise A |
| 3.2.1 | `src/A/AtoB.txt` | Input file for transformation A to B |
| 3.2.2 | `src/A/AtoC.txt` | Input file for transformation A to C |
| 3.3 | `src/OutArr/` | Intermediate output arrays |
| 3.3.1 | `src/OutArr/OutArrB.txt` | Output array B |
| 3.3.2 | `src/OutArr/OutArrC.txt` | Output array C |
| 3.4 | `src/Output/` | Final output files |
| 3.4.1 | `src/Output/Output_no_args.txt` | Output without arguments |
| 3.4.2 | `src/Output/Output8B.txt` | Output for 8B case |
| 3.4.3 | `src/Output/Output8C.txt` | Output for 8C case |
| 3.4.4 | `src/Output/Output512.txt` | Output for N=512 |
| 3.4.5 | `src/Output/Output1024.txt` | Output for N=1024 |
| 3.4.6 | `src/Output/Output10000.txt` | Output for N=10000 |
| 3.4.7 | `src/Output/Output20000.txt` | Output for N=20000 |
| 4 | `README.md` | Repository overview and usage instructions |

---

## Overview

The project utilizes the **CUDA architecture** to perform matrix calculations in parallel on a GPU. A random **N×N integer matrix** is generated and processed across CUDA threads for efficiency.

---

## Core Operations

- **Average Calculation (`calcAvg`)**: Computes the mean of all elements using **parallel reduction** and **atomic operations**.
- **Maximum Finding (`findMax`)**: Identifies the largest element in the matrix.
- **Matrix B Creation (`createB`)**:  
  Bᵢⱼ = a_max − Aᵢⱼ for i ≠ j  
  Bᵢᵢ = a_max  
  Also identifies the minimum element in B.
- **Matrix C Creation (`createC`)**:  
  Cᵢⱼ = 3·Aᵢⱼ + Aᵢ(j+1) + Aᵢ(j−1)  

---

## 3. Design & Implementation

**Optimization Techniques**:

- **Parallel Reductions**: Efficient aggregation for sum and maximum computations.
- **Shared Memory**: Reduces global memory latency by storing block-local data.
- **Synchronization**: Uses `__syncthreads()` to coordinate threads within a block.
- **Atomic Operations**: Implements `atomicMin` for floating-point numbers using `atomicCAS`.

---

# Installation & Setup Guide

This repository implements **parallel computations on 2D integer arrays** using **CUDA**, developed as part of the **Parallel Systems course** at the **University of West Attica**. The project demonstrates GPU-accelerated matrix operations using **CUDA threads, shared memory, and atomic operations**.

---

## Prerequisites

### Required Software
- **NVIDIA CUDA Toolkit** (≥ 11.0 recommended)  
  Download: [https://developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads)
- **NVIDIA GPU** with compute capability ≥ 3.0 (tested on NVIDIA TITAN RTX)
- **GCC compiler** (Linux/macOS) or compatible compiler on Windows
- **Make / Terminal** for compilation and execution

### Optional Software
- Text editor or IDE (VSCode, CLion, Nsight)  
- Spreadsheet viewer for performance analysis

---

## Installation Steps

### 1. Clone the Repository

```bash
git clone https://github.com/Parallel-Systems-aka-Uniwa/CUDA.git
```

Or download the ZIP archive and extract it.

### 2. Navigate to Project Directory
```bash
cd CUDA
```

Folder structure:
```bash
assign/
docs/
src/
README.md
```

`src/` contains CUDA source code (cuda1.cu) and input/output directories

`docs/` contains theory, exercises, and performance analysis

---

## Compilation Instructions
Compile the CUDA program using `nvcc`:
```bash
nvcc -o cuda1 src/cuda1.cu
```
Explanation:
- `-o cuda1` → output executable named cuda1
- `src/cuda1.cu` → CUDA source file

Ensure CUDA Toolkit paths are correctly set (`$PATH` and `$LD_LIBRARY_PATH` on Linux).

---

## Execution Instructions
Run the program with input matrix file and output file:
```bash
./cuda1 src/A/AtoB.txt src/OutArr/OutArrB.txt
```

Arguments:
- **Input file** → Path to input matrix (e.g., src/A/AtoB.txt)
- **Output file** → Path to store result matrix (e.g., src/OutArr/OutArrB.txt)

Example Runs
```bash
./cuda1 src/A/AtoB.txt src/OutArr/OutArrB.txt
./cuda1 src/A/AtoC.txt src/OutArr/OutArrC.txt
```
- Performs Matrix B or Matrix C operations based on input file
- Supports different matrix sizes (e.g., N=8, 512, 1024, 10000, 20000)

---

## Input Files
- Located in `src/A/`
- Files contain integer matrices in text format

Typical names:
- **AtoB.txt** → Input for Matrix B computation
- **AtoC.txt** → Input for Matrix C computation

---

## Output Files
Stored in `src/OutArr/` (intermediate arrays) or `src/Output/` (final results)

Examples:
- **OutArrB.txt** → Matrix B after computation
- **OutArrC.txt** → Matrix C after computation
- **Output512.txt** → Result for N=512
- **Output20000.txt** → Result for N=20000

---

## Core Operations
### Average (calcAvg)
Parallel reduction with atomic operations

### Maximum (findMax) 
Parallel search for largest element

### Matrix B (createB) 

$$
Bᵢⱼ = a_max − Aᵢⱼ (i ≠ j), Bᵢᵢ = a_max; 
$$

also finds min(B)

### Matrix C (createC) 

$$
Cᵢⱼ = 3·Aᵢⱼ + Aᵢ(j+1) + Aᵢ(j−1)
$$

---

## Performance Analysis

Experiments were conducted on various matrix sizes:

| Matrix Size | calcAvg (ms) | findMax (ms) | createB/C (ms) |
|------------|--------------|--------------|----------------|
| 8×8        | 0.204736     | 0.015552     | 0.015040 (B)   |
| 512×512    | 0.136576     | 0.016704     | 0.016576 (C)   |
| 1024×1024  | 36.310913    | 0.059424     | 0.072832 (B)   |
| 20000×20000| 39.388447    | 0.015104     | 0.011424 (C)   |

Observations:
- Execution times for calcAvg scale linearly with matrix size.
- `findMax` and `createC` remain high-performance even for large matrices due to optimized parallel kernels.