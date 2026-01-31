package yolo26

import (
	"fmt"
	"github.com/getcharzp/go-vision"
	ort "github.com/getcharzp/onnxruntime_purego"
	"github.com/up-zero/gotool/convertutil"
	"image"
)

// DetEngine YOLO26-det Engine
type DetEngine struct {
	session *ort.Session
	config  Config
}

// NewDetEngine 初始化检测引擎
func NewDetEngine(cfg Config) (*DetEngine, error) {
	oc := new(vision.OnnxConfig)
	_ = convertutil.CopyProperties(cfg, oc)

	if err := oc.New(); err != nil {
		return nil, fmt.Errorf("初始化失败: %w", err)
	}

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

	// Output Shape: [1, 300, 6]
	data, err := ort.GetTensorData[float32](outputValue)
	if err != nil {
		return nil, fmt.Errorf("获取输出数据失败: %w", err)
	}

	return e.postprocess(data, params), nil
}

// postprocess 后处理，输出结果解析
func (e *DetEngine) postprocess(data []float32, params imageParams) []DetResult {
	results := make([]DetResult, 0)

	const stride = 6
	numDetections := len(data) / stride

	for i := 0; i < numDetections; i++ {
		offset := i * stride

		// [x1, y1, x2, y2, score, class_id]
		x1 := data[offset+0]
		y1 := data[offset+1]
		x2 := data[offset+2]
		y2 := data[offset+3]
		score := data[offset+4]
		classID := int(data[offset+5])

		if score < e.config.ConfThreshold {
			continue
		}

		// 转换回原图坐标
		origX1 := max(0, int(x1/params.scale))
		origY1 := max(0, int(y1/params.scale))
		origX2 := min(params.origW, int(x2/params.scale))
		origY2 := min(params.origH, int(y2/params.scale))

		results = append(results, DetResult{
			ClassID: classID,
			Score:   score,
			Box:     image.Rect(origX1, origY1, origX2, origY2),
		})
	}

	return results
}
