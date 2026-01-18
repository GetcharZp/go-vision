package yolov11

import (
	"fmt"
	"github.com/getcharzp/go-vision"
	ort "github.com/getcharzp/onnxruntime_purego"
	"github.com/up-zero/gotool/convertutil"
	"image"
	"log"
)

// DetEngine YOLOv11-det Engine
type DetEngine struct {
	session *ort.Session
	config  Config
}

// NewDetEngine 初始化检测引擎
func NewDetEngine(cfg Config) (*DetEngine, error) {
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

	return &DetEngine{
		session: session,
		config:  cfg,
	}, nil
}

// Destroy 释放相关资源
func (e *DetEngine) Destroy() {
	if e.session != nil {
		e.session.Destroy()
	}
}

// Predict 执行检测推理
func (e *DetEngine) Predict(img image.Image) ([]DetResult, error) {
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
	outputValue := outputValues["output0"]
	defer outputValue.Destroy()

	// Output Shape: [1, 84, 8400]
	data, err := ort.GetTensorData[float32](outputValue)
	if err != nil {
		return nil, fmt.Errorf("获取输出数据失败: %w", err)
	}
	shape, err := outputValue.GetShape()
	if err != nil {
		return nil, fmt.Errorf("获取输出形状失败: %w", err)
	}

	// 后处理
	return e.postprocess(data, shape, params)
}

// postprocess 后处理
func (e *DetEngine) postprocess(data []float32, shape []int64, params imageParams) ([]DetResult, error) {
	numChannels := int(shape[1]) // 4 (box) + 80 (cls) = 84
	numAnchors := int(shape[2])  // 8400

	// 解析候选框
	candidates := e.parseCandidates(data, numChannels, numAnchors, params)
	// NMS
	keptIndices := nms(candidates, e.config.IOUThreshold)

	results := make([]DetResult, 0, len(keptIndices))
	for _, idx := range keptIndices {
		cand := candidates[idx]
		results = append(results, DetResult{
			ClassID: cand.classID,
			Score:   cand.score,
			Box:     cand.origBox,
		})
	}

	return results, nil
}

// parseCandidates 解析候选框
func (e *DetEngine) parseCandidates(data []float32, channels, anchors int, params imageParams) []candidate {
	var cands []candidate

	// 检查通道数
	expectedChannels := 4 + e.config.NumClasses
	if channels != expectedChannels {
		log.Printf("警告：传入的通道数(%d)与预期(%d)不匹配", channels, expectedChannels)
		return cands
	}

	for i := 0; i < anchors; i++ {
		// 找最大类别分数
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
			box:     [4]float32{x1, y1, x2, y2},
			origBox: image.Rect(origX1, origY1, origX2, origY2),
			score:   maxScore,
			classID: classID,
		})
	}
	return cands
}
