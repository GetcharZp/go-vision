package dinov2

import (
	"fmt"
	"image"
	"image/color"

	"github.com/getcharzp/go-vision"
	ort "github.com/getcharzp/onnxruntime_purego"
	"github.com/up-zero/gotool/convertutil"
)

// Engine DINOv2 分割引擎
type Engine struct {
	session *ort.Session
	config  Config
}

// NewEngine 初始化引擎
func NewEngine(cfg Config) (*Engine, error) {
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

	return &Engine{
		session: session,
		config:  cfg,
	}, nil
}

// Destroy 释放相关资源
func (e *Engine) Destroy() {
	if e.session != nil {
		e.session.Destroy()
	}
}

// Result 分割结果
type Result struct {
	Width  int // 原图宽度
	Height int // 原图高度

	// Labels 分割标签图, 每个像素值为对应的聚类编号 (0 ~ NumClusters-1)
	Labels *image.Gray
	// Masks 每个聚类的二值掩码, 白色(255) 表示属于该区域
	Masks []*image.Gray
}

// Predict 执行分割推理
//
// DINOv2 输出的是 Patch 特征 (last_hidden_state), 这里通过无监督聚类 (k-means)
// 将特征相似的区域划分到同一个分割区域, 从而得到逐像素的分割结果。
func (e *Engine) Predict(img image.Image) (*Result, error) {
	inputSize := e.config.InputSize
	if inputSize <= 0 {
		inputSize = 224
	}
	patchSize := e.config.PatchSize
	if patchSize <= 0 {
		patchSize = 14
	}
	numClusters := e.config.NumClusters
	if numClusters <= 0 {
		numClusters = 3
	}
	numIter := e.config.NumIter
	if numIter <= 0 {
		numIter = 30
	}

	// 预处理
	inputData, params := preprocess(img, inputSize)

	// 创建 Input Tensor
	inputTensor, err := ort.NewTensor([]int64{1, 3, int64(inputSize), int64(inputSize)}, inputData)
	if err != nil {
		return nil, fmt.Errorf("创建 Input Tensor 失败: %w", err)
	}
	defer inputTensor.Destroy()

	// 推理
	inputValues := map[string]*ort.Value{
		e.session.InputNames[0]: inputTensor,
	}
	outputs, err := e.session.Run(inputValues)
	if err != nil {
		return nil, fmt.Errorf("推理失败: %w", err)
	}
	output := outputs[e.session.OutputNames[0]]
	defer output.Destroy()

	// last_hidden_state: [1, numTokens, dim]
	hidden, err := ort.GetTensorData[float32](output)
	if err != nil {
		return nil, fmt.Errorf("获取输出数据失败: %w", err)
	}
	shape, err := output.GetShape()
	if err != nil {
		return nil, fmt.Errorf("获取输出形状失败: %w", err)
	}

	// 后处理: 聚类生成分割图
	return e.postprocess(hidden, shape, params, inputSize, patchSize, numClusters, numIter)
}

// postprocess 后处理, 生成分割结果
func (e *Engine) postprocess(hidden []float32, shape []int64, params imageParams, inputSize, patchSize, numClusters, numIter int) (*Result, error) {
	// 提取 Patch 特征 [grid*grid, dim]
	patchFeats, grid, dim, err := extractPatchFeatures(hidden, shape, patchSize, inputSize)
	if err != nil {
		return nil, fmt.Errorf("提取 Patch 特征失败: %w", err)
	}

	// 双线性上采样到输入分辨率 [inputSize, inputSize, dim]
	featMap := bilinearUpsample(patchFeats, grid, grid, dim, inputSize, inputSize)

	// L2 归一化
	l2normalize(featMap, dim)

	// k-means 聚类, 得到输入分辨率下的标签图
	numPixels := inputSize * inputSize
	labels := kmeans(featMap, numPixels, dim, numClusters, numIter)

	// 最近邻缩放到原图尺寸
	labelsResized := resizeLabels(labels, inputSize, inputSize, params.origW, params.origH)

	// 构建分割结果
	result := &Result{
		Width:  params.origW,
		Height: params.origH,
		Labels: image.NewGray(image.Rect(0, 0, params.origW, params.origH)),
		Masks:  make([]*image.Gray, numClusters),
	}
	for c := 0; c < numClusters; c++ {
		result.Masks[c] = image.NewGray(image.Rect(0, 0, params.origW, params.origH))
	}

	for idx, label := range labelsResized {
		x := idx % params.origW
		y := idx / params.origW

		result.Labels.SetGray(x, y, color.Gray{Y: uint8(label)})
		result.Masks[label].SetGray(x, y, color.Gray{Y: 255})
	}

	return result, nil
}
