package yolov11

import (
	"fmt"
	"github.com/getcharzp/go-vision"
	ort "github.com/getcharzp/onnxruntime_purego"
	"github.com/up-zero/gotool/convertutil"
	"image"
	"math"
)

// OBBEngine YOLOv11-OBB Engine
type OBBEngine struct {
	session *ort.Session
	config  Config
}

// NewOBBEngine 初始化 OBB 引擎
func NewOBBEngine(cfg Config) (*OBBEngine, error) {
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

	// Output Shaper: [1, 20, 21504]
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
func (e *OBBEngine) postprocess(data []float32, shape []int64, params imageParams) ([]OBBResult, error) {
	numChannels := int(shape[1])
	numAnchors := int(shape[2])

	// 解析候选框
	candidates := e.parseCandidates(data, numChannels, numAnchors, params)
	// NMS
	keptIndices := nms(candidates, e.config.IOUThreshold)

	results := make([]OBBResult, 0, len(keptIndices))
	for _, idx := range keptIndices {
		cand := candidates[idx]

		// 重新计算旋转后的 4 个角点
		corners := getRotatedCorners(cand.box[0], cand.box[1], cand.box[2], cand.box[3], cand.angle)

		// 映射回原图坐标
		origCorners := [4]image.Point{}
		for i, pt := range corners {
			ox := min(max(0, int(pt[0]/params.scale)), params.origW)
			oy := min(max(0, int(pt[1]/params.scale)), params.origH)

			origCorners[i] = image.Point{X: ox, Y: oy}
		}

		results = append(results, OBBResult{
			ClassID: cand.classID,
			Score:   cand.score,
			Corners: origCorners,
			Center:  image.Point{X: (origCorners[0].X + origCorners[2].X) / 2, Y: (origCorners[0].Y + origCorners[2].Y) / 2},
			Angle:   cand.angle,
		})
	}

	return results, nil
}

// parseCandidates 解析候选框
func (e *OBBEngine) parseCandidates(data []float32, channels, anchors int, params imageParams) []candidate {
	// data
	// T [cx, cy, w, h, c1...c15, angle]

	var cands []candidate
	angleIdx := channels - 1

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
		angle := data[angleIdx*anchors+i]

		// 获取旋转矩形的4个角点
		corners := getRotatedCorners(cx, cy, w, h, angle)

		// 找外接矩形的 min/max
		minX, minY := float32(math.MaxFloat32), float32(math.MaxFloat32)
		maxX, maxY := float32(-math.MaxFloat32), float32(-math.MaxFloat32)
		for _, pt := range corners {
			minX = min(pt[0], minX)
			maxX = max(pt[0], maxX)
			minY = min(pt[1], minY)
			maxY = max(pt[1], maxY)
		}
		origX1 := int(minX / params.scale)
		origY1 := int(minY / params.scale)
		origX2 := int(maxX / params.scale)
		origY2 := int(maxY / params.scale)

		cands = append(cands, candidate{
			box:     [4]float32{cx, cy, w, h},
			angle:   angle,
			origBox: image.Rect(origX1, origY1, origX2, origY2),
			score:   maxScore,
			classID: classID,
		})
	}
	return cands
}
