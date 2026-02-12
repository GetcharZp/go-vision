package yolo26

import (
	"fmt"
	"github.com/getcharzp/go-vision"
	ort "github.com/getcharzp/onnxruntime_purego"
	"github.com/up-zero/gotool/convertutil"
	"image"
)

// PoseEngine YOLO26-pose Engine
type PoseEngine struct {
	session *ort.Session
	config  Config
}

// NewPoseEngine 初始化姿态引擎
func NewPoseEngine(cfg Config) (*PoseEngine, error) {
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

	return &PoseEngine{
		session: session,
		config:  cfg,
	}, nil
}

// Destroy 释放相关资源
func (e *PoseEngine) Destroy() {
	if e.session != nil {
		e.session.Destroy()
	}
}

// Predict 执行姿态估计
func (e *PoseEngine) Predict(img image.Image) ([]PoseResult, error) {
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

	// Output Shaper: [1, 300, 57]
	data, err := ort.GetTensorData[float32](outputValue)
	if err != nil {
		return nil, fmt.Errorf("获取输出数据失败: %w", err)
	}

	// 后处理
	return e.postprocess(data, params)
}

// postprocess 后处理
func (e *PoseEngine) postprocess(data []float32, params imageParams) ([]PoseResult, error) {
	const (
		numObjects = 300
		attributes = 57 // 4(box) + 1(score) + 1(class) + 51(kpts)
	)

	results := make([]PoseResult, 0)

	for i := 0; i < numObjects; i++ {
		offset := i * attributes

		// 得分
		score := data[offset+4]
		if score < e.config.ConfThreshold {
			continue
		}

		// 类别
		classID := int(data[offset+5])

		// 边界框
		x1 := data[offset+0]
		y1 := data[offset+1]
		x2 := data[offset+2]
		y2 := data[offset+3]

		// 映射回原图尺寸
		origX1 := int(x1 / params.scale)
		origY1 := int(y1 / params.scale)
		origX2 := int(x2 / params.scale)
		origY2 := int(y2 / params.scale)

		rawKpts := data[offset+6 : offset+57]
		kpts := e.decodeKeyPoints(rawKpts, params)

		results = append(results, PoseResult{
			ClassID:   classID,
			Score:     score,
			Box:       image.Rect(origX1, origY1, origX2, origY2),
			KeyPoints: kpts,
		})
	}

	return results, nil
}

// decodeKeyPoints 关键点解码
func (e *PoseEngine) decodeKeyPoints(raw []float32, params imageParams) []KeyPoint {
	kpts := make([]KeyPoint, e.config.NumKeyPoints)

	for i := 0; i < e.config.NumKeyPoints; i++ {
		idx := i * 3
		x := raw[idx]
		y := raw[idx+1]
		conf := raw[idx+2]

		// 坐标映射回原图
		origX := min(max(0, int(x/params.scale)), params.origW)
		origY := min(max(0, int(y/params.scale)), params.origH)

		kpts[i] = KeyPoint{
			X:     origX,
			Y:     origY,
			Score: conf,
		}
	}
	return kpts
}
