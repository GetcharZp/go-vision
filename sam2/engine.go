package sam2

import (
	"fmt"
	"github.com/getcharzp/go-vision"
	"github.com/up-zero/gotool/convertutil"
	"github.com/up-zero/gotool/imageutil"
	"image"
	"runtime"

	ort "github.com/yalue/onnxruntime_go"
)

// Engine 持有 ONNX Session，负责创建 ImageContext
type Engine struct {
	encoderSession *ort.DynamicAdvancedSession
	decoderSession *ort.DynamicAdvancedSession
	config         Config
}

// NewEngine 初始化 sam2 引擎
func NewEngine(cfg Config) (*Engine, error) {
	onnxConfig := new(vision.OnnxConfig)
	if err := convertutil.CopyProperties(cfg, onnxConfig); err != nil {
		return nil, fmt.Errorf("复制参数失败: %w", err)
	}
	// 初始化 ONNX
	if err := onnxConfig.New(); err != nil {
		return nil, err
	}

	// encoder session
	encInputs := []string{"pixel_values"}
	encOutputs := []string{"image_embeddings.0", "image_embeddings.1", "image_embeddings.2"}
	encSession, err := ort.NewDynamicAdvancedSession(cfg.EncodeModelPath, encInputs, encOutputs, onnxConfig.SessionOptions)
	if err != nil {
		return nil, fmt.Errorf("创建 Encoder ONNX 会话失败: %w", err)
	}

	// decoder session
	decInputs := []string{
		"input_points", "input_labels", "input_boxes",
		"image_embeddings.0", "image_embeddings.1", "image_embeddings.2",
	}
	decOutputs := []string{"iou_scores", "pred_masks", "object_score_logits"}
	decSession, err := ort.NewDynamicAdvancedSession(cfg.DecodeModelPath, decInputs, decOutputs, onnxConfig.SessionOptions)
	if err != nil {
		encSession.Destroy()
		return nil, fmt.Errorf("创建 Decoder ONNX 会话失败: %w", err)
	}

	return &Engine{
		encoderSession: encSession,
		decoderSession: decSession,
		config:         cfg,
	}, nil
}

// Destroy 释放相关资源
func (e *Engine) Destroy() error {
	if e.encoderSession != nil {
		if err := e.encoderSession.Destroy(); err != nil {
			return fmt.Errorf("销毁 Encoder ONNX 会话失败: %w", err)
		}
	}
	if e.decoderSession != nil {
		if err := e.decoderSession.Destroy(); err != nil {
			return fmt.Errorf("销毁 Decoder ONNX 会话失败: %w", err)
		}
	}
	return nil
}

// ImageContext 包含特定图像的特征缓存和参数
type ImageContext struct {
	engine          *Engine
	imageEmbeddings []ort.Value

	origW, origH int
	scale        float32
	newW, newH   int
	isDestroyed  bool
}

// EncodeImage 图像特征提取
func (e *Engine) EncodeImage(img image.Image) (*ImageContext, error) {
	// 预处理
	bounds := img.Bounds()
	origW, origH := bounds.Dx(), bounds.Dy()

	scale := float32(inputSize) / float32(max(origW, origH))
	newW := int(float32(origW) * scale)
	newH := int(float32(origH) * scale)

	resizedImg := imageutil.Resize(img, newW, newH)
	tensorData := normalizeAndPad(resizedImg, inputSize, inputSize)

	// 创建 Input Tensor
	inputShape := ort.NewShape(1, 3, int64(inputSize), int64(inputSize))
	inputTensor, err := ort.NewTensor(inputShape, tensorData)
	if err != nil {
		return nil, fmt.Errorf("创建图片 Input Tensor 失败: %w", err)
	}
	defer inputTensor.Destroy()

	// Encoder 推理
	outputs := make([]ort.Value, 3)
	if err := e.encoderSession.Run([]ort.Value{inputTensor}, outputs); err != nil {
		return nil, fmt.Errorf("encoder 推理失败: %w", err)
	}

	ctx := &ImageContext{
		engine:          e,
		imageEmbeddings: outputs,
		origW:           origW,
		origH:           origH,
		scale:           scale,
		newW:            newW,
		newH:            newH,
	}

	// 设置 Finalizer 以防用户忘记 Destroy
	runtime.SetFinalizer(ctx, func(c *ImageContext) { c.Destroy() })

	return ctx, nil
}

// Destroy 释放图像特征缓存
func (ctx *ImageContext) Destroy() {
	if ctx.isDestroyed {
		return
	}
	for _, v := range ctx.imageEmbeddings {
		if v != nil {
			v.Destroy()
		}
	}
	ctx.imageEmbeddings = nil
	ctx.isDestroyed = true
}

// Result Mask 预测结果
type Result struct {
	Mask   []uint8 // 0 or 255
	Score  float32
	Width  int
	Height int
}

// DecodeRaw Mask解码并返回原始结果
func (ctx *ImageContext) DecodeRaw(points []Point) (*Result, error) {
	if ctx.isDestroyed {
		return nil, fmt.Errorf("图片特征已销毁")
	}

	// 坐标转换
	coords := make([]float32, 0, len(points)*2)
	labels := make([]int64, 0, len(points))

	for _, pt := range points {
		coords = append(coords, pt.X*ctx.scale, pt.Y*ctx.scale)
		labels = append(labels, int64(pt.Label))
	}

	numPoints := int64(len(points))

	// 准备 Decoder Tensors
	tPoints, err := ort.NewTensor(ort.NewShape(1, 1, numPoints, 2), coords)
	if err != nil {
		return nil, fmt.Errorf("创建 Decoder Points Tensor 失败: %w", err)
	}
	defer tPoints.Destroy()

	tLabels, err := ort.NewTensor(ort.NewShape(1, 1, numPoints), labels)
	if err != nil {
		return nil, fmt.Errorf("创建 Decoder Labels Tensor 失败: %w", err)
	}
	defer tLabels.Destroy()

	// box 通过 point 控制
	var emptyFloat []float32
	tBoxes, err := ort.NewTensor(ort.NewShape(1, 0, 4), emptyFloat)
	if err != nil {
		return nil, fmt.Errorf("创建 Decoder Boxes Tensor 失败: %w", err)
	}
	defer tBoxes.Destroy()

	inputs := []ort.Value{
		tPoints,
		tLabels,
		tBoxes,
		ctx.imageEmbeddings[0],
		ctx.imageEmbeddings[1],
		ctx.imageEmbeddings[2],
	}
	outputs := make([]ort.Value, 3)

	// Decoder 推理
	if err := ctx.engine.decoderSession.Run(inputs, outputs); err != nil {
		return nil, fmt.Errorf("decoder 推理失败: %w", err)
	}
	defer func() {
		for _, o := range outputs {
			o.Destroy()
		}
	}()

	// 获取最佳 Mask
	rawScores := outputs[0].(*ort.Tensor[float32]).GetData()
	rawMasks := outputs[1].(*ort.Tensor[float32]).GetData()

	bestIdx := 0
	bestScore := float32(-100.0)

	for i := 0; i < len(rawScores); i++ {
		if rawScores[i] > bestScore {
			bestScore = rawScores[i]
			bestIdx = i
		}
	}

	// 提取对应的 Mask Logits (256x256)
	pixelsPerMask := 256 * 256
	start := bestIdx * pixelsPerMask
	end := start + pixelsPerMask
	bestMaskLogits := rawMasks[start:end]

	validMaskW := int(float32(ctx.newW) / 4.0)
	validMaskH := int(float32(ctx.newH) / 4.0)

	finalMask := upscaleMaskLogits(bestMaskLogits, 256, validMaskW, validMaskH, ctx.origW, ctx.origH)

	return &Result{
		Mask:   finalMask,
		Score:  bestScore,
		Width:  ctx.origW,
		Height: ctx.origH,
	}, nil
}

// Decode Mask解码并返回图片
func (ctx *ImageContext) Decode(points []Point) (image.Image, float32, error) {
	result, err := ctx.DecodeRaw(points)
	if err != nil {
		return nil, 0, err
	}

	img := image.NewGray(image.Rect(0, 0, result.Width, result.Height))
	copy(img.Pix, result.Mask)
	return img, result.Score, nil
}
