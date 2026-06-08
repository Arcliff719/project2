## 1. 当前程序采用的算法

当前实现的是一个简化 JPEG 风格的有损图像压缩算法，核心流程如下：

1. 读取图像并转为灰度矩阵。
2. 将图像按边缘像素补齐到 8x8 的整数倍。
3. 对每个 8x8 图像块做二维 DCT（离散余弦变换）。
4. 使用 JPEG 亮度量化表，并根据 `quality` 参数缩放量化强度。
5. 对量化后的 DCT 系数做 Zigzag 扫描。
6. 对 DC 系数使用 DPCM 差分编码。
7. 对 AC 系数使用 RLE 游程编码。
8. 对 DC/AC 符号分别建立 Huffman 编码表。
9. 将压缩数据打包为 bitstream，供后续信道编码与 BSC/BEC 信道仿真使用。
10. 解码时执行 Huffman 解码、RLE/DPCM 逆变换、反 Zigzag、反量化、IDCT，恢复图像。

需要注意：当前 `main.py` 默认把图片转为灰度图处理，因此输出重建图是灰度图。`src/source_coding.py` 中已经保留了 RGB 分通道接口，后续如果需要彩色图压缩，可以在主程序里接入 `encode_image_rgb()` 和 `decode_image_rgb()`。

## 2. 当前已经实现的内容

- 图像读取：支持 `png/jpg/jpeg/bmp/tiff/tif/pgm/ppm`。
- 批量处理：默认读取 `graph/` 下所有图片。
- 单图处理：可通过 `--image` 指定一张图片。
- 质量参数：`--quality 1..100`，数值越大，恢复质量越高，压缩率越低。
- 压缩 bitstream 输出：保存为 `bitstream_qXX.bin`。
- 重建图输出：保存为 `reconstructed_qXX.png`。
- 质量评估：计算 PSNR、MSE、压缩比、BPP。
- 复杂度评估：统计编码/解码耗时和内存峰值。
- 可视化分析：输出 rate-distortion 曲线、复杂度曲线、压缩性能汇总图。
- 边界测试：已修复纯色图导致 Huffman 单符号树解码崩溃的问题。

## 3. 图像数据集

图片都放在 `graph/` 目录中

如果需要重新生成测试图像，运行：

```powershell
python generate_test_graphs.py
```

## 4. 如何运行

在项目根目录运行：

```powershell
cd D:\AI_Project\project
```

处理 `graph/` 目录中的全部图片：

```powershell
python main.py --quality 50 --output output
```

运行后每张图片会打印类似下面的格式：

```text
Source Coding Demo - Quality = 50
Quality parameter: 50
Image size    : 512x512
Original bits : 2,097,152
Compressed    : 148,040 bits (18,505.0 bytes)
Compression   : 14.17x
Bits/pixel    : 0.5647
PSNR          : 37.49 dB
MSE           : 11.5951
Reconstructed -> output\graph_bar_chart\reconstructed_q50.png
Bitstream     -> output\graph_bar_chart\bitstream_q50.bin (18505 bytes)
```

处理单张图片：

```powershell
python main.py --image graph\graph_bar_chart.png --quality 50 --output output_test
```

对单张图片测试多个质量参数，并生成分析图：

```powershell
python main.py --image graph\graph_bar_chart.png --output output_test_range
```

未指定 `--quality` 时，程序会自动测试默认质量参数：

```text
Quality parameters: [1, 5, 10, 25, 50, 75, 90, 100]
```

并打印每个质量参数下的 PSNR、压缩比、BPP、编码/解码耗时，最后仍然保留并生成以下分析图：

```text
rate_distortion.png
complexity.png
compression_summary.png
```

## 5. 如何调试与测试

基础测试：

```powershell
python tests\test_source_coding.py
```

该测试会检查：

- DCT/IDCT 是否能正确往返。
- 纯色图是否能正常编码解码。
- 非 8 倍数尺寸图片是否能正确补齐和裁剪。
- `graph/` 中真实图片是否能正常压缩、恢复并得到合理 PSNR。

语法检查：

```powershell
python -m compileall main.py src tests
```

快速人工检查：

```powershell
python main.py --image graph\graph_bar_chart.png --quality 5 --output output_q5
python main.py --image graph\graph_bar_chart.png --quality 90 --output output_q90
```

然后对比 `output_q5/` 和 `output_q90/` 里的重建图。一般来说，`quality=5` 压缩率高但失真明显，`quality=90` 失真小但 bitstream 更大。
## 6. 文件结构

```text
project/
  main.py                    # 主程序入口
  generate_test_graphs.py     # 测试图像生成脚本
  graph/                      # 图像集合
  src/
    source_coding.py          # A 分工核心：DCT 压缩/解压/Huffman/bitstream
    analysis.py               # PSNR、MSE、复杂度、曲线图
  tests/
    test_source_coding.py     # 基础测试
  README.md                   # 当前中文交接说明
```

## 7. 当前限制

- 默认主程序只处理灰度图，所以颜色会丢失。
- Huffman 树目前保存在 `EncodedImage` 对象中，没有单独序列化进 `.bin` 文件。
- A 部分只负责信源编码；B 分支已新增 BSC/BEC 信道模型与 `none`/重复码/Hamming(7,4) 信道编码。
- 如果后续要独立保存并跨程序读取 bitstream，需要额外设计文件格式，把 Huffman 表、量化表和图像尺寸一起保存。

## 8. B 分工：信道编码与 BSC/BEC 信道模拟

本分支新增 `src/channel_coding.py`，用于承接 A 同学输出的压缩 `bitstream`，完成“信道编码 -> 信道传输 -> 信道译码 -> 交回信源解码器”的流程。

### 已实现的信道模型

- **BSC（二进制对称信道）**：每个编码后比特以概率 `p` 独立翻转。
- **BEC（二进制擦除信道）**：每个编码后比特以概率 `p` 独立变成擦除符号 `-1`。
- 支持设置随机种子 `--seed`，便于复现实验结果。

### 已实现的信道编码

- `none`：不加保护，直接传输，用作 baseline。
- `repetition`：奇数重复码，默认重复 3 次，通过多数投票恢复；BEC 下会忽略擦除位。
- `hamming`：Hamming(7,4) 线性分组码，默认推荐；可纠正每个 7-bit 码字中的 1 个 BSC 翻转错误，也可恢复每个码字中的 1 个 BEC 擦除。

### B 分工命令示例

对一张图像运行信源编码，并在 BSC/BEC 上测试 Hamming(7,4)：

```powershell
python main.py --image graph\graph_bar_chart.png --quality 50 --output output_channel --simulate-channel --channel both --channel-code hamming --error-probs 0.01,0.05,0.1
```

只测试 BSC，并使用 3 重重复码：

```powershell
python main.py --image graph\graph_bar_chart.png --quality 50 --output output_channel_rep --simulate-channel --channel bsc --channel-code repetition --repetition-factor 3 --error-probs 0.01,0.05,0.1
```

运行后，每个图像输出目录会额外生成：

```text
channel_results_hamming.csv       # BER、冗余率、纠错数、不可纠错块数、运行时间、恢复 PSNR
channel_bsc_hamming_p0p01.png     # 经过指定信道后恢复的图像
channel_bec_hamming_p0p01.png
...
```

### B 分工测试

```powershell
python tests\test_channel_coding.py
python -m compileall main.py src tests
```

`tests/test_channel_coding.py` 会检查：

- bytes 与 bit vector 的转换是否无损。
- Hamming(7,4) 是否能纠正单个 BSC 翻转错误。
- Hamming(7,4) 是否能恢复单个 BEC 擦除。
- BSC/BEC 随机仿真是否可通过 seed 复现。
- `p=0` 时完整 bitstream 是否可无误传输并交回 A 同学的信源解码器。
