package sam2

import "github.com/getcharzp/go-vision"

type Label int

const (
	LabelBackground  Label = 0 // 背景/排除
	LabelForeground  Label = 1 // 前景/点击
	LabelBoxTopLeft  Label = 2 // 框选左上
	LabelBoxBotRight Label = 3 // 框选右下
)

// 均值和方差常量
const (
	MeanG = 0.456
	MeanB = 0.406
	MeanR = 0.485

	StdG = 0.224
	StdB = 0.225
	StdR = 0.229
)

const (
	// inputSize 输入图片的长边尺寸
	inputSize = 1024
	// maskThreshold 阈值
	maskThreshold = 0.0
)

type Point struct {
	X, Y  float32
	Label Label
}

// Config 配置项
type Config struct {
	// 必填参数
	OnnxRuntimeLibPath string // onnxruntime.dll (或 .so, .dylib) 的路径
	EncodeModelPath    string // 图片特征提取模型
	DecodeModelPath    string // Mask解码模型

	// 可选参数
	UseCuda    bool // (可选) 是否启用 CUDA
	NumThreads int  // (可选) ONNX 线程数, 默认由CPU核心数决定
}

// DefaultConfig 返回默认配置
func DefaultConfig() Config {
	return Config{
		OnnxRuntimeLibPath: vision.DefaultLibraryPath(),
		EncodeModelPath:    "./sam2_weights/vision_encoder.onnx",
		DecodeModelPath:    "./sam2_weights/prompt_encoder_mask_decoder.onnx",
	}
}
