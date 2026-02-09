package yolo26

import (
	ort "github.com/getcharzp/onnxruntime_purego"
	"github.com/up-zero/gotool/imageutil"
	"image"
	"math"
)

// preprocess 预处理
func preprocess(img image.Image, inputSize int) (*ort.Value, imageParams, error) {
	bounds := img.Bounds()
	params := imageParams{
		origW: bounds.Dx(),
		origH: bounds.Dy(),
	}

	scale := float32(inputSize) / float32(max(params.origW, params.origH))
	params.scale = scale

	newW := int(float32(params.origW) * scale)
	newH := int(float32(params.origH) * scale)

	resized := imageutil.Resize(img, newW, newH)

	// 准备 Tensor 数据 (CHW + Normalize 0-1)
	data := make([]float32, 3*inputSize*inputSize)
	for y := 0; y < newH; y++ {
		for x := 0; x < newW; x++ {
			r, g, b, _ := resized.At(x, y).RGBA()

			idx := y*inputSize + x
			data[idx] = float32(r) / 65535.0                       // R
			data[inputSize*inputSize+idx] = float32(g) / 65535.0   // G
			data[2*inputSize*inputSize+idx] = float32(b) / 65535.0 // B
		}
	}

	tensor, err := ort.NewTensor([]int64{1, 3, int64(inputSize), int64(inputSize)}, data)
	return tensor, params, err
}

func sigmoid(x float32) float32 {
	return 1.0 / (1.0 + float32(math.Exp(float64(-x))))
}
