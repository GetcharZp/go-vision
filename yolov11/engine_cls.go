package yolov11

import (
	"fmt"
	"github.com/getcharzp/go-vision"
	"github.com/up-zero/gotool/convertutil"
	ort "github.com/yalue/onnxruntime_go"
	"image"
	"sort"
)

// ClsEngine YOLOv11-cls Engine
type ClsEngine struct {
	session *ort.DynamicAdvancedSession
	config  Config
}

// NewClsEngine 初始化分类引擎
func NewClsEngine(cfg Config) (*ClsEngine, error) {
	onnxConfig := new(vision.OnnxConfig)
	if err := convertutil.CopyProperties(cfg, onnxConfig); err != nil {
		return nil, fmt.Errorf("复制参数失败: %w", err)
	}
	// 初始化 ONNX
	if err := onnxConfig.New(); err != nil {
		return nil, err
	}

	// 创建 Session
	inputs := []string{"images"}
	outputs := []string{"output0"}
	session, err := ort.NewDynamicAdvancedSession(cfg.ModelPath, inputs, outputs, onnxConfig.SessionOptions)
	if err != nil {
		return nil, fmt.Errorf("创建 ONNX 会话失败: %w", err)
	}

	return &ClsEngine{
		session: session,
		config:  cfg,
	}, nil
}

// Destroy 释放相关资源
func (e *ClsEngine) Destroy() {
	if e.session != nil {
		e.session.Destroy()
	}
}

// Predict 执行分类推理
//
// # Params:
//
//	img: 待分类图片
//	topK: 指定返回概率最高的 K 个类别
func (e *ClsEngine) Predict(img image.Image, topK int) ([]ClassResult, error) {
	// 预处理
	inputTensor, _, err := preprocess(img, e.config.InputSize)
	if err != nil {
		return nil, fmt.Errorf("预处理失败: %w", err)
	}
	defer inputTensor.Destroy()

	// 推理
	outputValues := make([]ort.Value, 1)
	err = e.session.Run([]ort.Value{inputTensor}, outputValues)
	if err != nil {
		return nil, fmt.Errorf("推理失败: %w", err)
	}
	defer outputValues[0].Destroy()

	// Output Shape: [1, 1000]
	rawOutput := outputValues[0].(*ort.Tensor[float32])
	data := rawOutput.GetData()

	// 后处理
	return e.postprocess(data, topK), nil
}

// postprocess 后处理
func (e *ClsEngine) postprocess(logits []float32, topK int) []ClassResult {
	// 转换为分类结果
	results := make([]ClassResult, len(logits))
	for i, score := range logits {
		results[i] = ClassResult{
			ClassID: i,
			Score:   score,
		}
	}

	sort.Slice(results, func(i, j int) bool {
		return results[i].Score > results[j].Score
	})

	return results[:min(topK, len(results))]
}
