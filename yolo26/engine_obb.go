package yolo26

import (
	"fmt"
	"github.com/getcharzp/go-vision"
	ort "github.com/getcharzp/onnxruntime_purego"
	"github.com/up-zero/gotool/convertutil"
	"image"
	"math"
)

// OBBEngine YOLO26-OBB Engine
type OBBEngine struct {
	session *ort.Session
	config  Config
}

// NewOBBEngine 初始化 OBB 引擎
func NewOBBEngine(cfg Config) (*OBBEngine, error) {
	oc := new(vision.OnnxConfig)
	_ = convertutil.CopyProperties(cfg, oc)

	// 初始化 ONNX
	if err := oc.New(); err != nil {
		return nil, err
	}

	// 创建 Session
	session, err := oc.OnnxEngine.NewSession(cfg.ModelPath, oc.SessionOptions)
	if err != nil {
		return nil, fmt.Errorf("failed to load obb model %s: %w", cfg.ModelPath, err)
	}

	return &OBBEngine{
		session: session,
		config:  cfg,
	}, nil
}

// Destroy 释放相关资源
func (e *OBBEngine) Destroy() {
	if e.session != nil {
		e.session.Destroy()
	}
}

// Predict 执行旋转目标检测
func (e *OBBEngine) Predict(img image.Image) ([]OBBResult, error) {
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

	// 解析输出 [1, 300, 7]
	data, err := ort.GetTensorData[float32](outputValue)
	if err != nil {
		return nil, fmt.Errorf("获取输出数据失败: %w", err)
	}

	shape, err := outputValue.GetShape()
	if err != nil || len(shape) < 3 {
		return nil, fmt.Errorf("输出形状异常")
	}

	numBoxes := int(shape[1]) // 300
	numAttrs := int(shape[2]) // 7

	results := make([]OBBResult, 0)

	for i := 0; i < numBoxes; i++ {
		offset := i * numAttrs

		// [cx, cy, w, h, score, class_id, angle]
		cx := data[offset+0]
		cy := data[offset+1]
		w := data[offset+2]
		h := data[offset+3]
		score := data[offset+4]
		classID := int(data[offset+5])
		angle := data[offset+6]

		if score < e.config.ConfThreshold {
			continue
		}

		// 获取旋转矩形的4个角点
		corners := getRotatedCorners(cx, cy, w, h, angle)

		// 映射回原图坐标
		var origCorners [4]image.Point
		for j, pt := range corners {
			// 边界检查
			ox := min(max(0, int(math.Round(float64(pt[0]/params.scale)))), params.origW)
			oy := min(max(0, int(math.Round(float64(pt[1]/params.scale)))), params.origH)
			origCorners[j] = image.Point{X: ox, Y: oy}
		}

		results = append(results, OBBResult{
			ClassID: classID,
			Score:   score,
			Corners: origCorners,
			Center: image.Point{
				X: (origCorners[0].X + origCorners[2].X) / 2,
				Y: (origCorners[0].Y + origCorners[2].Y) / 2,
			},
			Angle: angle,
		})
	}

	return results, nil
}
