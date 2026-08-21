package dinov2

import (
	"fmt"
	"image"
	"math"

	"github.com/up-zero/gotool/imageutil"
)

// ImageNet 归一化常量 (DINOv2 使用 ImageNet 的 mean/std 进行预处理)
const (
	MeanR float32 = 0.485
	MeanG float32 = 0.456
	MeanB float32 = 0.406
	StdR  float32 = 0.229
	StdG  float32 = 0.224
	StdB  float32 = 0.225
)

// imageParams 图片尺寸信息
type imageParams struct {
	origW, origH int
}

// preprocess 预处理: 缩放到 inputSize×inputSize 并按 ImageNet mean/std 归一化
func preprocess(img image.Image, inputSize int) ([]float32, imageParams) {
	bounds := img.Bounds()
	params := imageParams{
		origW: bounds.Dx(),
		origH: bounds.Dy(),
	}

	resized := imageutil.Resize(img, inputSize, inputSize)

	// 准备 Tensor 数据 (CHW + ImageNet Normalize)
	data := make([]float32, 3*inputSize*inputSize)
	for y := 0; y < inputSize; y++ {
		for x := 0; x < inputSize; x++ {
			r, g, b, _ := resized.At(x, y).RGBA()

			idx := y*inputSize + x
			data[idx] = (float32(r)/65535.0 - MeanR) / StdR                       // R
			data[inputSize*inputSize+idx] = (float32(g)/65535.0 - MeanG) / StdG   // G
			data[2*inputSize*inputSize+idx] = (float32(b)/65535.0 - MeanB) / StdB // B
		}
	}
	return data, params
}

// extractPatchFeatures 提取 Patch 特征 (去掉 CLS token)
//
// # Params:
//
//	hidden: last_hidden_state 输出 [numTokens, dim]
//	shape:  输出张量的形状 [1, numTokens, dim]
//	patchSize: 图像 Patch 尺寸
//	inputSize: 模型输入尺寸
//
// # Returns:
//
//	features: [grid*grid, dim] 的 Patch 特征
//	grid: 特征图的边长
//	dim:  特征维度
func extractPatchFeatures(hidden []float32, shape []int64, patchSize, inputSize int) ([]float32, int, int, error) {
	numTokens := int(shape[1])
	dim := int(shape[2])
	grid := inputSize / patchSize
	patches := grid * grid

	// 自动判断是否包含 CLS token (通常在索引 0)
	startIdx := 0
	if numTokens == patches+1 {
		startIdx = 1
	}
	if numTokens-startIdx != patches {
		return nil, 0, 0, fmt.Errorf("输出 token 数量不匹配: got %d, 期望 %d", numTokens, patches+startIdx)
	}

	features := make([]float32, patches*dim)
	copy(features, hidden[startIdx*dim:])
	return features, grid, dim, nil
}

// bilinearUpsample 将 [srcH, srcW, dim] 的特征图双线性插值放大到 [dstH, dstW, dim]
func bilinearUpsample(src []float32, srcW, srcH, dim, dstW, dstH int) []float32 {
	dst := make([]float32, dstW*dstH*dim)

	for oy := 0; oy < dstH; oy++ {
		// 源坐标 (align_corners=False 的映射方式)
		sy := (float32(oy)+0.5)*float32(srcH)/float32(dstH) - 0.5
		sy = clampF(sy, 0, float32(srcH-1))
		y0 := int(sy)
		y1 := min(y0+1, srcH-1)
		fy := sy - float32(y0)

		for ox := 0; ox < dstW; ox++ {
			sx := (float32(ox)+0.5)*float32(srcW)/float32(dstW) - 0.5
			sx = clampF(sx, 0, float32(srcW-1))
			x0 := int(sx)
			x1 := min(x0+1, srcW-1)
			fx := sx - float32(x0)

			// 双线性插值 4 个源特征
			s00 := (y0*srcW + x0) * dim
			s01 := (y0*srcW + x1) * dim
			s10 := (y1*srcW + x0) * dim
			s11 := (y1*srcW + x1) * dim

			dIdx := (oy*dstW + ox) * dim
			for d := 0; d < dim; d++ {
				v00 := src[s00+d]
				v01 := src[s01+d]
				v10 := src[s10+d]
				v11 := src[s11+d]

				top := v00*(1-fx) + v01*fx
				bot := v10*(1-fx) + v11*fx
				dst[dIdx+d] = top*(1-fy) + bot*fy
			}
		}
	}
	return dst
}

// l2normalize 对每个特征向量做 L2 归一化
func l2normalize(features []float32, dim int) {
	for i := 0; i < len(features); i += dim {
		var sum float32
		for j := 0; j < dim; j++ {
			v := features[i+j]
			sum += v * v
		}
		norm := float32(math.Sqrt(float64(sum)))
		if norm > 1e-8 {
			for j := 0; j < dim; j++ {
				features[i+j] /= norm
			}
		}
	}
}

// resizeLabels 将标签图从 [srcW, srcH] 最近邻缩放到 [dstW, dstH] (映射回原图)
func resizeLabels(labels []int, srcW, srcH, dstW, dstH int) []int {
	dst := make([]int, dstW*dstH)
	for oy := 0; oy < dstH; oy++ {
		srcY := min(int(float32(oy)*float32(srcH)/float32(dstH)), srcH-1)
		for ox := 0; ox < dstW; ox++ {
			srcX := min(int(float32(ox)*float32(srcW)/float32(dstW)), srcW-1)
			dst[oy*dstW+ox] = labels[srcY*srcW+srcX]
		}
	}
	return dst
}

// clampF 将 v 限制在 [lo, hi]
func clampF(v, lo, hi float32) float32 {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}
