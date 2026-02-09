package yolo26

import (
	"fmt"
	"github.com/getcharzp/go-vision"
	ort "github.com/getcharzp/onnxruntime_purego"
	"github.com/up-zero/gotool/convertutil"
	"image"
	"image/color"
)

// SegEngine YOLO26-seg Engine
type SegEngine struct {
	session *ort.Session
	config  Config
}

// NewSegEngine 初始化分割引擎
func NewSegEngine(cfg Config) (*SegEngine, error) {
	oc := new(vision.OnnxConfig)
	_ = convertutil.CopyProperties(cfg, oc)
	// 初始化 ONNX
	if err := oc.New(); err != nil {
		return nil, err
	}

	// 创建 Session
	session, err := oc.OnnxEngine.NewSession(cfg.ModelPath, oc.SessionOptions)
	if err != nil {
		return nil, fmt.Errorf("创建 ONNX 会话失败: %w", err)
	}

	return &SegEngine{
		session: session,
		config:  cfg,
	}, nil
}

// Destroy 释放相关资源
func (e *SegEngine) Destroy() {
	if e.session != nil {
		e.session.Destroy()
	}
}

// Predict 执行分割推理
func (e *SegEngine) Predict(img image.Image) ([]SegResult, error) {
	// 预处理
	inputTensor, params, err := preprocess(img, e.config.InputSize)
	if err != nil {
		return nil, fmt.Errorf("预处理失败: %w", err)
	}
	defer inputTensor.Destroy()

	// 推理
	inputValues := map[string]*ort.Value{
		"images": inputTensor,
	}
	outputValues, err := e.session.Run(inputValues)
	if err != nil {
		return nil, fmt.Errorf("推理失败: %w", err)
	}
	defer func() {
		for _, v := range outputValues {
			v.Destroy()
		}
	}()

	// output0: Detections [1,300,38]
	// output1: Mask Protos [1, 32, 160, 160]

	// 后处理
	return e.postprocess(outputValues["output0"], outputValues["output1"], params)
}

// postprocess 后处理
func (e *SegEngine) postprocess(out0, out1 *ort.Value, params imageParams) ([]SegResult, error) {
	data0, err := ort.GetTensorData[float32](out0)
	if err != nil {
		return nil, fmt.Errorf("获取数据失败: %w", err)
	}
	data1, err := ort.GetTensorData[float32](out1)
	if err != nil {
		return nil, fmt.Errorf("获取 Mask Protos 失败: %w", err)
	}
	shape1, _ := out1.GetShape()
	protoC, protoH, protoW := int(shape1[1]), int(shape1[2]), int(shape1[3])

	const (
		numDetections = 300
		dimTotal      = 38
		indexMask     = 6
	)

	var results []SegResult

	// [x1, y1, x2, y2, score, class, mask_coeffs...]
	for i := 0; i < numDetections; i++ {
		offset := i * dimTotal

		score := data0[offset+4]
		if score < e.config.ConfThreshold {
			continue
		}

		x1 := data0[offset+0]
		y1 := data0[offset+1]
		x2 := data0[offset+2]
		y2 := data0[offset+3]
		classID := int(data0[offset+5])

		// 映射回原图
		origX1 := max(0, int(x1/params.scale))
		origY1 := max(0, int(y1/params.scale))
		origX2 := min(params.origW, int(x2/params.scale))
		origY2 := min(params.origH, int(y2/params.scale))
		origBox := image.Rect(origX1, origY1, origX2, origY2)

		// 提取 32 位 Mask 系数
		coeffs := data0[offset+indexMask : offset+indexMask+32]

		// 解码 Mask
		mask := e.decodeMask(origBox, coeffs, data1, protoC, protoH, protoW, params)

		results = append(results, SegResult{
			ClassID: classID,
			Score:   score,
			Box:     origBox,
			Mask:    mask,
		})
	}

	return results, nil
}

// decodeMask Mask解码
//
// # Params:
//
//	origBox: 原图上的检测框
//	maskCoeffs: 当前检测框对应的 32 个 Mask 系数
//	protos: 模型 Output1 的原型数据
//	c, h, w: 原型数据的维度 (32, 160, 160)
//	params: 图片尺寸缩放参数
func (e *SegEngine) decodeMask(origBox image.Rectangle, maskCoeffs []float32, protos []float32, c, h, w int, params imageParams) *image.Gray {
	finalMask := image.NewGray(image.Rect(0, 0, params.origW, params.origH))

	// Mask 原型图相对于 InputSize(640) 的缩放比例
	maskStride := float32(e.config.InputSize) / float32(w)

	for y := origBox.Min.Y; y < origBox.Max.Y; y++ {
		for x := origBox.Min.X; x < origBox.Max.X; x++ {
			// 映射回 640 尺度
			inputX := float32(x) * params.scale
			inputY := float32(y) * params.scale

			// 映射回 160 Mask 尺度
			mx := int(inputX / maskStride)
			my := int(inputY / maskStride)

			if mx >= 0 && mx < w && my >= 0 && my < h {
				// 计算该像素的 Mask 值 (Dot Product)
				sum := float32(0.0)
				for k := 0; k < c; k++ {
					sum += maskCoeffs[k] * protos[k*h*w+my*w+mx]
				}

				if sigmoid(sum) > e.config.MaskThreshold {
					finalMask.SetGray(x, y, color.Gray{Y: 255})
				}
			}
		}
	}
	return finalMask
}
