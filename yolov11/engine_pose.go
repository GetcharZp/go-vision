package yolov11

import (
	"fmt"
	"github.com/getcharzp/go-vision"
	ort "github.com/getcharzp/onnxruntime_purego"
	"github.com/up-zero/gotool/convertutil"
	"image"
	"log"
)

// PoseEngine YOLOv11-pose Engine
type PoseEngine struct {
	session *ort.Session
	config  Config
}

// NewPoseEngine 初始化姿态引擎
func NewPoseEngine(cfg Config) (*PoseEngine, error) {
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

	// Output Shaper: [1, 56, 8400]
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
func (e *PoseEngine) postprocess(data []float32, shape []int64, params imageParams) ([]PoseResult, error) {
	numChannels := int(shape[1])
	numAnchors := int(shape[2])

	// 解析候选框
	candidates := e.parseCandidates(data, numChannels, numAnchors, params)
	// NMS
	keptIndices := nms(candidates, e.config.IOUThreshold)

	results := make([]PoseResult, 0, len(keptIndices))
	for _, idx := range keptIndices {
		cand := candidates[idx]
		// 关键点解码
		kpts := e.decodeKeyPoints(cand.rawKeyPoints, params)

		results = append(results, PoseResult{
			ClassID:   cand.classID,
			Score:     cand.score,
			Box:       cand.origBox,
			KeyPoints: kpts,
		})
	}

	return results, nil
}

// parseCandidates 解析候选框
func (e *PoseEngine) parseCandidates(data []float32, channels, anchors int, params imageParams) []candidate {
	// data
	// T [cx, cy, w, h, c1, x1,y1,conf1...x17,y17,conf17]

	var cands []candidate

	// 检查通道数
	expectedChannels := 4 + e.config.NumClasses + e.config.NumKeyPoints*3
	if channels != expectedChannels {
		log.Printf("警告：传入的通道数(%d)与预期(%d)不匹配", channels, expectedChannels)
		return cands
	}

	kptStartIdx := 4 + e.config.NumClasses
	// 每个关键点包含 3 个 float: x, y, conf
	numKptValues := e.config.NumKeyPoints * 3

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

		// 转换回原图矩形坐标
		x1 := cx - w/2
		y1 := cy - h/2
		x2 := cx + w/2
		y2 := cy + h/2
		origX1 := int(x1 / params.scale)
		origY1 := int(y1 / params.scale)
		origX2 := int(x2 / params.scale)
		origY2 := int(y2 / params.scale)

		// 暂存 Raw KeyPoints 数据
		rawKpts := make([]float32, numKptValues)
		for k := 0; k < numKptValues; k++ {
			rawKpts[k] = data[(kptStartIdx+k)*anchors+i]
		}

		cands = append(cands, candidate{
			box:          [4]float32{x1, y1, x2, y2},
			origBox:      image.Rect(origX1, origY1, origX2, origY2),
			score:        maxScore,
			classID:      classID,
			rawKeyPoints: rawKpts,
		})
	}
	return cands
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
