package dinov2

import (
	"github.com/getcharzp/go-vision"
)

// Config 引擎的初始化参数
type Config struct {
	ModelPath          string // ONNX 模型路径
	OnnxRuntimeLibPath string // ONNX Runtime 动态库路径

	// 推理参数
	InputSize   int // 模型输入尺寸, 默认 224 (DINOv2 常用 224 / 518)
	NumClusters int // 分割区域(聚类)数量, 默认 3
	NumIter     int // k-means 最大迭代次数, 默认 30

	// 模型参数
	PatchSize int // 图像 Patch 尺寸, 默认 14 (dinov2-base)

	// 可选参数
	UseCuda           bool // (可选) 是否启用 CUDA
	NumThreads        int  // (可选) ONNX 线程数, 默认由CPU核心数决定
	EnableCpuMemArena bool // (可选) 是否开启 ONNX 内存池
}

// DefaultConfig 默认配置
func DefaultConfig() Config {
	return Config{
		OnnxRuntimeLibPath: vision.DefaultLibraryPath(),
		ModelPath:          "./dinov2_weights/model.onnx",
		InputSize:          224,
		NumClusters:        3,
		NumIter:            30,
		PatchSize:          14,
	}
}
