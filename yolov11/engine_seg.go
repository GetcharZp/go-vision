package yolov11

import (
	"fmt"
	"github.com/getcharzp/go-vision"
	ort "github.com/getcharzp/onnxruntime_purego"
	"github.com/up-zero/gotool/convertutil"
	"image"
	"image/color"
	"log"
)

// SegEngine YOLOv11-seg Engine
type SegEngine struct {
	session *ort.Session
	config  Config
}

// NewSegEngine 初始化分割引擎
func NewSegEngine(cfg Config) (*SegEngine, error) {
	oc := new(vision.OnnxConfig)
	if err := convertutil.CopyProperties(cfg, oc); err != nil {
		return nil, fmt.Errorf("复制参数失败: %w", err)
	}
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

	// output0: Detections [1, 116, 8400]
	// output1: Mask Protos [1, 32, 160, 160]

	// 后处理
	return e.postprocess(outputValues["output0"], outputValues["output1"], params)
}

// postprocess 后处理
func (e *SegEngine) postprocess(out0, out1 *ort.Value, params imageParams) ([]SegResult, error) {
	data0, err := ort.GetTensorData[float32](out0)
	if err != nil {
		return nil, fmt.Errorf("获取输出数据失败: %w", err)
	}
	shape0, err := out0.GetShape() // [1, 116, 8400]
	if err != nil {
		return nil, fmt.Errorf("获取输出形状失败: %w", err)
	}
	numChannels := int(shape0[1]) // 4 (box) + 80 (cls) + 32 (mask) = 116
	numAnchors := int(shape0[2])  // 8400

	data1, err := ort.GetTensorData[float32](out1)
	if err != nil {
		return nil, fmt.Errorf("获取输出数据失败: %w", err)
	}
	shape1, err := out1.GetShape() // [1, 32, 160, 160]
	if err != nil {
		return nil, fmt.Errorf("获取输出形状失败: %w", err)
	}
	protoC, protoH, protoW := int(shape1[1]), int(shape1[2]), int(shape1[3])

	// 解析候选框
	candidates := e.parseCandidates(data0, numChannels, numAnchors, params)
	// NMS
	keptIndices := nms(candidates, e.config.IOUThreshold)

	results := make([]SegResult, 0, len(keptIndices))

	// 生成 Mask
	for _, idx := range keptIndices {
		cand := candidates[idx]

		// 生成二进制 mask
		mask := e.decodeMask(cand, data1, protoC, protoH, protoW, params)
		results = append(results, SegResult{
			ClassID: cand.classID,
			Score:   cand.score,
			Box:     cand.origBox, // image.Rectangle
			Mask:    mask,
		})
	}

	return results, nil
}

// parseCandidates 解析候选框
//
// # Params:
//
//	data: 模型输出的数组
//		[x1, x2 ..., x8400]
//		[y1, y2 ..., y8400]
//		[w1, w2 ..., w8400]
//		[h1, h2 ..., h8400]
//		[c1_1, c1_2 ..., c1_8400]
//		[c2_1, c2_2 ..., c2_8400]
//		...
//		[m1, m2 ..., m8400]
//	channels: 模型输出的通道数
//	anchors: 模型输出的锚点数
//	params: 图片尺寸信息
func (e *SegEngine) parseCandidates(data []float32, channels, anchors int, params imageParams) []candidate {
	var cands []candidate

	// 检查通道数
	expectedChannels := 4 + e.config.NumClasses + e.config.NumMaskCoeffs
	if channels != expectedChannels {
		log.Printf("警告：传入的通道数(%d)与预期(%d)不匹配", channels, expectedChannels)
		return cands
	}

	for i := 0; i < anchors; i++ {
		// 找最大分类分数
		maxScore := float32(0.0)
		classID := -1
		for c := 0; c < e.config.NumClasses; c++ {
			score := data[(4+c)*anchors+i]
			if score > maxScore {
				maxScore = score
				classID = c
			}
		}
		if maxScore < e.config.ConfThreshold {
			continue
		}

		// 提取坐标
		cx := data[0*anchors+i]
		cy := data[1*anchors+i]
		w := data[2*anchors+i]
		h := data[3*anchors+i]

		// 提取 Mask 系数
		coeffs := make([]float32, e.config.NumMaskCoeffs)
		for j := 0; j < e.config.NumMaskCoeffs; j++ {
			coeffs[j] = data[(4+e.config.NumClasses+j)*anchors+i]
		}

		// 转换回原图矩形坐标
		x1 := cx - w/2
		y1 := cy - h/2
		x2 := cx + w/2
		y2 := cy + h/2
		origX1 := max(0, int(x1/params.scale))
		origY1 := max(0, int(y1/params.scale))
		origX2 := min(params.origW, int(x2/params.scale))
		origY2 := min(params.origH, int(y2/params.scale))

		cands = append(cands, candidate{
			box:        [4]float32{x1, y1, x2, y2},
			origBox:    image.Rect(origX1, origY1, origX2, origY2),
			score:      maxScore,
			classID:    classID,
			maskCoeffs: coeffs,
		})
	}
	return cands
}

// decodeMask Mask解码
//
// # Params:
//
//	cand: 候选结果
//	protos: 模型输出的 Mask 原型图
//	c: 原型掩码的通道数
//	h: 单个原型掩码的高度
//	w: 单个原型掩码的宽度
//	params: 图片尺寸信息
func (e *SegEngine) decodeMask(cand candidate, protos []float32, c, h, w int, params imageParams) *image.Gray {
	finalMask := image.NewGray(image.Rect(0, 0, params.origW, params.origH))

	// Mask 原型图相对于 InputSize(640) 的缩放比例
	maskStride := float32(e.config.InputSize) / float32(w)

	// 遍历原图上的 Box 区域
	origBox := cand.origBox
	coeffs := cand.maskCoeffs
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
					sum += coeffs[k] * protos[k*h*w+my*w+mx]
				}

				if sigmoid(sum) > e.config.MaskThreshold {
					finalMask.SetGray(x, y, color.Gray{Y: 255})
				}
			}
		}
	}
	return finalMask
}
